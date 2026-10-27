import logging
import time
from datetime import date
from decimal import Decimal
from celery import shared_task
from django.db import transaction
from django.utils import timezone

from apps.instruments.models import Instrument
from apps.marketdata.services import MarketDataService
from apps.marketdata.models import Bar, LatestQuote
from apps.strategies.models import Strategy, StrategyVersion, Signal
from apps.strategies.engines import get_engine
from apps.portfolios.models import Portfolio
from apps.trading.models import RebalanceRun, OrderIntent
from apps.trading.allocator import PortfolioAllocator
from apps.trading.execution import OrderDispatcher
from apps.trading.guards import skip_if_not_trading_day
from apps.trading.pipeline import (
    get_or_create_today_pipeline, pipeline_step, record_timing, step_lock, step_skip_reason, update_pipeline,
)
from apps.accounting.models import LogicalPosition, EquitySnapshot
from apps.accounting.services import portfolio_equity
from apps.brokers.services import sync_alpaca_account
from core.locks import redis_lock
from core.market_time import get_ny_now

logger = logging.getLogger(__name__)


def sanitize_json_for_db(val):
    """
    Recursively sanitize objects for PostgreSQL JSON/JSONB.
    Replaces NaN, Inf, and -Inf with None (null in JSON).
    Converts numpy numeric types and Decimals to standard Python types.
    """
    import math
    import numpy as np
    from decimal import Decimal

    if val is None:
        return None
    if isinstance(val, dict):
        return {str(k): sanitize_json_for_db(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [sanitize_json_for_db(v) for v in val]
    if isinstance(val, (float, np.floating)):
        return None if (math.isnan(val) or math.isinf(val)) else float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, Decimal):
        return float(val) if val.is_finite() else None
    if isinstance(val, bool):
        return val
    try:
        import pandas as pd
        if pd.isna(val):
            return None
    except Exception:
        pass
    return str(val)


@shared_task(name="marketdata_refresh", queue="marketdata")
def marketdata_refresh():
    """Descarga barras incrementales y LatestQuote de todos los instrumentos activos."""
    logger.info("Executing marketdata_refresh...")
    with step_lock("marketdata") as acquired:
        skip = step_skip_reason("marketdata", acquired, force=True)
        if skip:
            logger.info(skip)
            return skip
        return _marketdata_refresh()


def _marketdata_refresh():
    with pipeline_step("marketdata") as metrics:
        svc = MarketDataService()
        instruments = Instrument.objects.filter(tradable=True)
        errors = 0
        for inst in instruments:
            # Historial completo la primera vez (lo exigen apalancamiento y regímenes); luego incremental.
            svc.ensure_history(inst)
            if svc.update_latest_quote(inst.symbol) is None:
                errors += 1
        count = instruments.count()
        metrics.update({"instruments": count, "errors": errors})
    return f"Refreshed {count} instruments"


def get_frozen_ranking(version, as_of_date):
    """
    Devuelve el orden de ranking vigente (símbolos del puesto 1 al N) de un bot ranked_allocation,
    o None si hoy toca recalcularlo: no hay ranking previo, es fecha de rebalanceo de algún
    portafolio activo que usa el bot, o ese portafolio tiene un rebalanceo manual pendiente.
    """
    from apps.strategies.models import AllocationSnapshot
    from apps.trading.rebalance_policy import is_periodic_rebalance_date

    last_snapshot = AllocationSnapshot.objects.filter(version=version).order_by("-as_of").first()
    if not last_snapshot:
        return None

    portfolios = Portfolio.objects.filter(
        status="active", allocations__strategy=version.strategy, allocations__enabled=True,
    ).distinct()
    for p in portfolios:
        if p.rebalance_pending or is_periodic_rebalance_date(p, as_of_date):
            return None

    ranked = sorted(
        ((d.get("rank"), sym) for sym, d in (last_snapshot.diagnostics or {}).items() if isinstance(d, dict)),
        key=lambda item: (item[0] is None, item[0]),
    )
    if not ranked or any(rank is None for rank, _ in ranked):
        return None
    return [sym for _, sym in ranked]


@shared_task(name="compute_regimes_and_signals", queue="signals")
@skip_if_not_trading_day("compute_regimes_and_signals")
def compute_regimes_and_signals(force: bool = False):
    """
    Calcula régimen macro y corre compute_signals para cada versión de estrategia activa.
    Sin `force` se omite si el paso de hoy ya está `done` o `running`; nunca corren dos a la vez.
    """
    logger.info("Executing compute_regimes_and_signals...")
    with step_lock("signals") as acquired:
        skip = step_skip_reason("signals", acquired, force)
        if skip:
            logger.info(skip)
            return skip
        return _compute_regimes_and_signals()


def _compute_regimes_and_signals():
    with pipeline_step("signals") as metrics:
        per_version = {}
        versions = StrategyVersion.objects.filter(status="live", strategy__is_active=True).select_related("strategy")
        signals_computed = 0
        today_ny = get_ny_now().date()

        for v in versions:
            version_started = time.monotonic()
            strat = v.strategy
            engine = get_engine(strat.engine)

            instruments = v.instruments.all().select_related("instrument")
            ohlcv_dict = {}
            weights = {}

            for si in instruments:
                sym = si.instrument.symbol
                weights[sym] = si.weight

                # Auto-warmup: garantiza el historial completo disponible antes de calcular señales,
                # para que EMAs, Z-Scores y el Profit Factor walk-forward usen todos los trades históricos.
                if not si.instrument.full_history_synced:
                    MarketDataService().ensure_history(si.instrument, incremental=False)

                rows = list(
                    Bar.objects.filter(instrument=si.instrument).order_by("date")
                    .values_list("date", "open", "high", "low", "close", "volume")
                )
                if rows:
                    import pandas as pd
                    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")
                    ohlcv_dict[sym] = df.astype(float)

            if not ohlcv_dict:
                continue

            # Los bots cross_asset guardan el activo de señales como instrumento con rol signal_source
            engine_params = dict(v.params or {})
            if not engine_params.get("source_symbol"):
                source_si = next((si for si in instruments if si.role == "signal_source"), None)
                if source_si:
                    engine_params["source_symbol"] = source_si.instrument.symbol

            # El ranking de los bots ranked_allocation solo se recalcula en fecha de rebalanceo del
            # portafolio; entre fechas se reutiliza el orden del último snapshot.
            if strat.kind == "ranked_allocation":
                frozen_ranking = get_frozen_ranking(v, today_ny)
                if frozen_ranking:
                    engine_params["frozen_ranking"] = frozen_ranking

            try:
                targets, diagnostics = engine.compute_signal(ohlcv_dict, engine_params, weights)

                # For ranked_allocation bots, audit the ranking into AllocationSnapshot
                if strat.kind == "ranked_allocation":
                    from apps.strategies.models import AllocationSnapshot
                    AllocationSnapshot.objects.create(
                        version=v,
                        as_of=timezone.now(),
                        weights={sym: str(targets.get(sym, Decimal("0.0"))) for sym in targets},
                        diagnostics=sanitize_json_for_db(diagnostics),
                    )

                for sym, exposure in targets.items():
                    inst = Instrument.objects.filter(symbol=sym).first()
                    if inst:
                        last_sig = Signal.objects.filter(version=v, instrument=inst).order_by("-as_of").first()
                        changed = (last_sig.target_exposure != exposure) if last_sig else True
                        clean_diag = sanitize_json_for_db(diagnostics.get(sym, {}))
                        Signal.objects.create(
                            version=v,
                            instrument=inst,
                            as_of=timezone.now(),
                            target_exposure=exposure,
                            changed=changed,
                            diagnostics=clean_diag,
                        )

                        signals_computed += 1
            except Exception as e:
                logger.error(f"Error computing signal for {strat.slug} v{v.version}: {e}")
            per_version[f"{strat.slug}@v{v.version}"] = round(time.monotonic() - version_started, 3)

        metrics.update({"versions": len(per_version), "signals": signals_computed, "per_version": per_version})

    return f"Computed {signals_computed} signals"


@shared_task(name="orchestrate_portfolio_runs", queue="pipeline")
@skip_if_not_trading_day("orchestrate_portfolio_runs")
def orchestrate_portfolio_runs(force: bool = False):
    """
    Evalúa rebalanceo y genera targets e intents para portafolios activos.
    Sin `force` se omite si el paso de hoy ya está `done` o `running`; nunca corren dos a la vez.
    """
    logger.info("Executing orchestrate_portfolio_runs...")
    with step_lock("orchestration") as acquired:
        skip = step_skip_reason("orchestration", acquired, force)
        if skip:
            logger.info(skip)
            return skip
        return _orchestrate_portfolio_runs()


def claim_scheduled_run(portfolio: Portfolio, trade_date: date):
    """
    Crea el RebalanceRun programado del día para el portafolio, o devuelve None si ya existe uno
    (planned/executing/done). La fila del portafolio se bloquea con select_for_update y el chequeo
    se repite dentro de la misma transacción: en Postgres esto serializa a dos procesos que
    orquestan a la vez y es la garantía real contra runs (y órdenes) duplicados. El redis_lock del
    llamador solo evita trabajo repetido (sin Redis cae a `True` y no excluye nada; SQLite ignora
    select_for_update pero serializa las escrituras).
    El equity se fija después, fuera de la transacción, para no sostener el bloqueo durante HTTP.
    """
    with transaction.atomic():
        Portfolio.objects.select_for_update().get(pk=portfolio.pk)
        already_run = RebalanceRun.objects.filter(
            portfolio=portfolio,
            as_of__date=trade_date,
            trigger="scheduled",
            status__in=["planned", "executing", "done"],
        ).exists()
        if already_run:
            return None
        return RebalanceRun.objects.create(
            portfolio=portfolio,
            trigger="scheduled",
            as_of=timezone.now(),
            status="planned",
        )


def _orchestrate_portfolio_runs():
    with pipeline_step("orchestration") as metrics:
        portfolios = Portfolio.objects.filter(status="active", trading_enabled=True)
        created_runs = 0
        failed = []
        today_ny = get_ny_now().date()

        for p in portfolios:
            with redis_lock(f"orchestrate:{p.id}:{today_ny}", timeout=300, blocking_timeout=1) as acquired:
                if not acquired:
                    logger.info(f"Portfolio '{p.name}' (id={p.id}) lo está orquestando otro proceso. Omitiendo.")
                    continue

                # Idempotencia: evitar corridas duplicadas en el mismo día para el mismo portafolio
                run = claim_scheduled_run(p, today_ny)
                if run is None:
                    logger.info(f"Portfolio '{p.name}' (id={p.id}) ya fue rebalanceado hoy ({today_ny}). Omitiendo.")
                    continue

                try:
                    # Dimensionar sobre el capital real: saldo del broker recién sincronizado
                    # (o contabilidad interna sin broker).
                    broker = p.broker_accounts.filter(trading_enabled=True).first()
                    if broker and broker.provider == "alpaca":
                        sync_alpaca_account(broker)
                    equity = portfolio_equity(p)
                    run.equity_base = equity["equity"]
                    run.decisions = {"equity_source": equity["source"]}
                    run.save(update_fields=["equity_base", "decisions"])

                    allocator = PortfolioAllocator(run)
                    intents = allocator.generate_targets_and_intents()
                    if p.rebalance_pending:
                        p.rebalance_pending = False
                        p.save(update_fields=["rebalance_pending"])
                    if intents:
                        run.status = "executing"
                        run.save(update_fields=["status"])
                    else:
                        run.status = "done"
                        run.save(update_fields=["status"])
                    created_runs += 1
                except Exception as e:
                    # El run queda `failed`: el despacho lo ignora y el fallback puede reintentarlo.
                    logger.error(f"Orquestación falló para portafolio '{p.name}' (id={p.id}): {e}")
                    run.status = "failed"
                    run.save(update_fields=["status"])
                    failed.append(f"{p.id}: {type(e).__name__}: {e}")

        metrics.update({"portfolios": portfolios.count(), "runs": created_runs, "failed": len(failed)})
        if failed:
            raise RuntimeError(f"Orquestación falló en {len(failed)} portafolio(s): {'; '.join(failed)}")

    return f"Orchestrated {created_runs} portfolio runs"


def dispatch_pending_intent(dispatcher: OrderDispatcher, intent_id: int) -> bool:
    """
    Despacha un intent solo si sigue `pending`, re-leyéndolo con select_for_update dentro de una
    transacción: dos despachos concurrentes nunca envían el mismo intent (el segundo lo ve
    `dispatched`/`cancelled` y lo omite). El client_order_id = idempotency_key sigue protegiendo
    del lado de Alpaca. Los errores del envío se capturan dentro de la transacción para conservar
    los estados que fija el dispatcher (p. ej. cancelled + missed_cutoff). True si se despachó.
    """
    with transaction.atomic():
        intent = (
            # of=("self",): Postgres no admite FOR UPDATE sobre el lado nulable del join (broker_account)
            OrderIntent.objects.select_for_update(of=("self",))
            .select_related("run__portfolio", "broker_account", "instrument")
            .get(pk=intent_id)
        )
        if intent.status != "pending":
            logger.info(f"Intent {intent_id} ya está {intent.status}: lo despachó otro proceso. Omitiendo.")
            return False
        try:
            dispatcher.dispatch_intent(intent)
            return True
        except Exception as e:
            logger.error(f"Failed to dispatch intent {intent_id}: {e}")
            return False


@shared_task(name="dispatch_moc_and_manual_orders", queue="execution")
@skip_if_not_trading_day("dispatch_moc_and_manual_orders")
def dispatch_moc_and_manual_orders():
    """
    Despacho (15:57 NY): envía órdenes de mercado DAY (admiten fracciones) a Alpaca y genera
    los fills simulados + avisos de Telegram. El nombre de la tarea se conserva por compatibilidad
    con los schedules ya registrados; ya no se usan órdenes MOC.
    """
    logger.info("Executing dispatch_moc_and_manual_orders...")
    started = time.monotonic()
    active_runs = RebalanceRun.objects.filter(status="executing")
    dispatcher = OrderDispatcher()
    dispatched_count = 0

    for run in active_runs:
        for intent_id in list(run.intents.filter(status="pending").values_list("id", flat=True)):
            if dispatch_pending_intent(dispatcher, intent_id):
                dispatched_count += 1
        # No pisar estados terminales fijados durante el despacho (p. ej. missed_cutoff)
        run.refresh_from_db(fields=["status"])
        if run.status == "executing":
            run.status = "done"
            run.save(update_fields=["status"])

    try:
        record_timing("dispatch", {"seconds": round(time.monotonic() - started, 3), "orders": dispatched_count})
    except Exception as e:
        logger.error(f"No se pudo registrar la métrica de despacho: {e}")
    return f"Dispatched {dispatched_count} orders"


@shared_task(name="run_daily_trading_pipeline", queue="pipeline")
@skip_if_not_trading_day("run_daily_trading_pipeline")
def run_daily_trading_pipeline():
    """
    Cadena de preparación (15:50 NY):
    1. Descarga cotizaciones y barras (marketdata_refresh)
    2. Calcula regímenes y señales cuantitativas (compute_regimes_and_signals)
    3. Rebalancea portafolios generando intenciones de orden (orchestrate_portfolio_runs)

    El envío de órdenes NO ocurre aquí: lo hace dispatch_moc_and_manual_orders a las 15:57 NY,
    para ejecutar lo más cerca posible del cierre con órdenes de mercado DAY.
    """
    logger.info(">>> [PIPELINE 15:50 NY] Iniciando pipeline automatizado consecutivo...")
    trade_date = get_ny_now().date()

    def mark_started(run):
        run.status = "running"
        run.started_at = timezone.now()
        run.finished_at = None

    update_pipeline(trade_date, mark_started)
    try:
        r1 = marketdata_refresh()
        logger.info(f"[1/3] Market data refresh: {r1}")
    except Exception as e:
        logger.error(f"[1/3] Market data refresh falló: {e}")

    try:
        r2 = compute_regimes_and_signals()
        logger.info(f"[2/3] Regimes & signals: {r2}")
    except Exception as e:
        logger.error(f"[2/3] Compute signals falló: {e}")

    try:
        r3 = orchestrate_portfolio_runs()
        logger.info(f"[3/3] Portfolio rebalance: {r3}")
    except Exception as e:
        logger.error(f"[3/3] Portfolio rebalance falló: {e}")

    def mark_finished(run):
        steps_ok = all(run.step_status(step) == "done" for step in run.STEPS)
        run.status = "done" if steps_ok else "failed"
        run.finished_at = timezone.now()

    update_pipeline(trade_date, mark_finished)
    return "Daily trading pipeline completed successfully."


@shared_task(name="fallback_compute_signals", queue="pipeline")
@skip_if_not_trading_day("fallback_compute_signals")
def fallback_compute_signals():
    """
    Respaldo programado (15:52 NY) del paso de señales: solo corre si el paso de hoy está
    `pending` o `failed`; si el pipeline ya lo terminó o lo está corriendo, se omite.
    """
    status = get_or_create_today_pipeline().signals_status
    if status in ("done", "running"):
        msg = f"Skipped: signals already {status} today"
        logger.info(f"[FALLBACK señales] {msg}")
        return msg
    logger.warning(f"[FALLBACK señales] El paso de señales está '{status}'. Ejecutando respaldo.")
    return compute_regimes_and_signals()


@shared_task(name="fallback_orchestrate", queue="pipeline")
@skip_if_not_trading_day("fallback_orchestrate")
def fallback_orchestrate():
    """
    Respaldo programado (15:53 NY) de la orquestación: solo corre si el paso de hoy está
    `pending` o `failed`. Si las señales de hoy no quedaron `done`, avisa por Telegram que el
    rebalanceo usará las últimas señales disponibles (no actualizadas) y continúa, como hasta ahora.
    """
    from apps.notifications.telegram import send_to_telegram

    run = get_or_create_today_pipeline()
    if run.orchestration_status in ("done", "running"):
        msg = f"Skipped: orchestration already {run.orchestration_status} today"
        logger.info(f"[FALLBACK orquestación] {msg}")
        return msg
    if run.signals_status != "done":
        send_to_telegram(
            f"⚠️ [PIPELINE {run.trade_date}] Rebalanceo con señales no actualizadas: el paso de señales "
            f"está '{run.signals_status}'. La orquestación de respaldo usará las últimas señales disponibles."
        )
    logger.warning(f"[FALLBACK orquestación] El paso de orquestación está '{run.orchestration_status}'. Ejecutando respaldo.")
    return orchestrate_portfolio_runs()


def escape_markdown(text: str) -> str:
    """Escapa los caracteres de Markdown de Telegram para textos con nombres de pasos o errores."""
    for ch in ("\\", "_", "*", "`", "["):
        text = text.replace(ch, "\\" + ch)
    return text


@shared_task(name="pipeline_watchdog", queue="celery")
@skip_if_not_trading_day("pipeline_watchdog")
def pipeline_watchdog():
    """
    Watchdog (15:56 NY, antes del despacho de 15:57): si el pipeline de hoy no terminó (`done`),
    avisa por Telegram con el estado y la duración de cada paso y el tiempo transcurrido.
    """
    from apps.notifications.telegram import send_to_telegram
    from apps.trading.models import DailyPipelineRun

    today = get_ny_now().date()
    run = DailyPipelineRun.objects.filter(trade_date=today).first()
    if run and run.status == "done":
        return "Pipeline OK"

    if not run:
        lines = [f"🚨 [PIPELINE {today}] No arrancó: no hay registro del pipeline a las 15:56 NY."]
    else:
        lines = [f"🚨 [PIPELINE {today}] No terminó a tiempo (estado: {run.status})."]
        if run.started_at:
            elapsed = (timezone.now() - run.started_at).total_seconds()
            lines.append(f"Transcurrido desde el inicio: {elapsed:.0f} s")
        for step in DailyPipelineRun.STEPS:
            seconds = (run.timings or {}).get(step, {}).get("seconds")
            duration = f" ({seconds:.1f} s)" if seconds is not None else ""
            lines.append(f"- {step}: {run.step_status(step)}{duration}")
        for err in (run.errors or [])[-3:]:
            lines.append(f"Error en {err.get('step')}: {err.get('error')}")
    send_to_telegram(escape_markdown("\n".join(lines)))
    return "Alert sent"


@shared_task(name="pipeline_post_cutoff_check", queue="celery")
@skip_if_not_trading_day("pipeline_post_cutoff_check")
def pipeline_post_cutoff_check():
    """
    Chequeo post-corte (16:01 NY): avisa por Telegram si hoy hay RebalanceRun en `missed_cutoff`
    o intents que siguen `pending` (no se enviaron antes del corte).
    """
    from apps.notifications.telegram import send_to_telegram

    today = get_ny_now().date()
    missed = list(
        RebalanceRun.objects.filter(as_of__date=today, status="missed_cutoff")
        .select_related("portfolio").values_list("id", "portfolio__name")
    )
    pending = list(
        OrderIntent.objects.filter(run__as_of__date=today, status="pending")
        .select_related("instrument", "run__portfolio")
        .values_list("id", "instrument__symbol", "run__portfolio__name")
    )
    if not missed and not pending:
        return "Post-cutoff OK"

    lines = [f"🚨 [POST-CORTE {today}] Revisión de órdenes de hoy:"]
    for run_id, portfolio_name in missed:
        lines.append(f"- Run #{run_id} ({portfolio_name}): missed_cutoff")
    if pending:
        lines.append(f"Intents sin enviar: {len(pending)}")
        for intent_id, symbol, portfolio_name in pending[:10]:
            lines.append(f"- Intent #{intent_id} {symbol} ({portfolio_name})")
    send_to_telegram(escape_markdown("\n".join(lines)))
    return f"Alert sent: {len(missed)} missed_cutoff runs, {len(pending)} pending intents"


@shared_task(name="reconcile_broker_fills", queue="execution")
@skip_if_not_trading_day("reconcile_broker_fills")
def reconcile_broker_fills():
    """
    Reconciliación post-mercado (16:05 NY):
    Consulta a Alpaca el estado real de todas las órdenes enviadas hoy,
    recupera precios promedio de ejecución (fill_price) y cantidades reales (qty),
    crea los registros Fill, actualiza posiciones (LogicalPosition) y refresca
    el balance de capital real en la cuenta del broker antes del snapshot diario.
    """
    from apps.trading.models import Order, Fill
    from alpaca.trading.client import TradingClient

    logger.info("Executing reconcile_broker_fills post-market (16:05 NY)...")
    pending_orders = Order.objects.filter(
        source="broker",
        status__in=["submitted", "accepted", "partially_filled"],
    ).exclude(broker_order_id="").select_related("portfolio", "broker_account", "instrument", "intent__run")

    dispatcher = OrderDispatcher()
    reconciled_fills = 0
    synced_accounts = set()

    for order in pending_orders:
        account = order.broker_account
        if not account or not account.credential:
            continue

        creds = account.credential.get_secrets()
        api_key = creds.get("key_id") or creds.get("api_key")
        api_secret = creds.get("secret_key") or creds.get("secret")
        is_paper = account.environment == "paper"

        if not api_key or not api_secret:
            continue

        try:
            client = TradingClient(api_key, api_secret, paper=is_paper)
            alp_order = client.get_order_by_id(order.broker_order_id)
            alp_status = str(alp_order.status).lower()

            if "filled" in alp_status:
                filled_qty = Decimal(str(alp_order.filled_qty or order.qty))
                fill_price = Decimal(str(alp_order.filled_avg_price or "0.00"))

                fill, created = Fill.objects.get_or_create(
                    order=order,
                    defaults={
                        "fill_price": fill_price,
                        "qty": filled_qty,
                        "commission": Decimal("0.00"),
                        "source": "broker_alpaca",
                        "filled_at": getattr(alp_order, "filled_at", None) or timezone.now(),
                    }
                )

                if created:
                    dispatcher._record_fill_bookkeeping(
                        fill=fill,
                        intent=order.intent,
                        instrument=order.instrument,
                        side=order.side,
                        qty=filled_qty,
                        price=fill_price
                    )
                    reconciled_fills += 1

                order.status = "filled"
                order.save(update_fields=["status"])
                logger.info(f"Reconciled fill for order {order.client_order_id}: {filled_qty} @ ${fill_price}")

            elif any(s in alp_status for s in ["cancel", "expired", "rejected"]):
                order.status = "cancelled" if "cancel" in alp_status else "rejected"
                order.save(update_fields=["status"])

            # Sincronizar el saldo real de la cuenta de Alpaca
            if account.id not in synced_accounts:
                alp_acc = client.get_account()
                if alp_acc and hasattr(alp_acc, "equity") and alp_acc.equity is not None:
                    account.managed_capital = Decimal(str(alp_acc.equity))
                    account.save(update_fields=["managed_capital"])
                    synced_accounts.add(account.id)
                    logger.info(f"Synced managed_capital for {account.display_name}: ${account.managed_capital}")

        except Exception as e:
            logger.error(f"Error reconciling order {order.id} ({order.broker_order_id}): {e}")

    return f"Reconciled {reconciled_fills} fills across {len(synced_accounts)} broker accounts"


POSITION_RECON_TOLERANCE = Decimal("0.0001")


def diff_positions(logical: dict, broker: dict, tolerance: Decimal = POSITION_RECON_TOLERANCE) -> list:
    """
    Compara cantidades por símbolo entre la contabilidad lógica y el broker.
    Devuelve [(symbol, logical_qty, broker_qty)] para cada símbolo que difiera más que la tolerancia.
    """
    mismatches = []
    for symbol in sorted(set(logical) | set(broker)):
        logical_qty = logical.get(symbol, Decimal("0"))
        broker_qty = broker.get(symbol, Decimal("0"))
        if abs(logical_qty - broker_qty) > tolerance:
            mismatches.append((symbol, logical_qty, broker_qty))
    return mismatches


@shared_task(name="reconcile_positions_vs_broker", queue="execution")
@skip_if_not_trading_day("reconcile_positions_vs_broker")
def reconcile_positions_vs_broker():
    """
    Red de seguridad post-cierre (16:10 NY): compara la suma de LogicalPosition por instrumento
    con las posiciones reales de cada cuenta Alpaca y alerta por Telegram si difieren.
    Solo informa; no corrige posiciones ni envía órdenes.
    """
    from django.db.models import Sum
    from alpaca.trading.client import TradingClient
    from apps.brokers.models import BrokerAccount
    from apps.notifications.telegram import send_to_telegram

    logger.info("Executing reconcile_positions_vs_broker...")
    accounts = BrokerAccount.objects.filter(
        provider="alpaca", execution_mode="auto", trading_enabled=True,
    ).select_related("credential", "portfolio")

    checked = 0
    accounts_with_mismatch = 0

    for account in accounts:
        if not account.credential:
            continue
        creds = account.credential.get_secrets()
        api_key = creds.get("key_id") or creds.get("api_key")
        api_secret = creds.get("secret_key") or creds.get("secret")
        if not api_key or not api_secret:
            continue

        try:
            client = TradingClient(api_key, api_secret, paper=account.environment == "paper")
            broker_qty = {p.symbol: Decimal(str(p.qty)) for p in client.get_all_positions()}
        except Exception as e:
            logger.error(f"Error fetching Alpaca positions for {account.display_name}: {e}")
            continue

        logical_rows = (
            LogicalPosition.objects.filter(broker_account=account)
            .values("instrument__symbol")
            .annotate(total=Sum("qty"))
        )
        logical_qty = {r["instrument__symbol"]: r["total"] for r in logical_rows if r["total"]}

        checked += 1
        mismatches = diff_positions(logical_qty, broker_qty)
        if mismatches:
            accounts_with_mismatch += 1
            lines = [
                f"🚨 [DESCUADRE DE POSICIONES] Portafolio: {account.portfolio.name}",
                f"Broker: {account.display_name}",
            ]
            for symbol, l_qty, b_qty in mismatches:
                lines.append(f"- {symbol}: lógico {l_qty} vs broker {b_qty} (dif {b_qty - l_qty})")
            send_to_telegram("\n".join(lines))
            logger.warning(f"Position mismatch for {account.display_name}: {mismatches}")

    return f"Checked {checked} broker accounts, {accounts_with_mismatch} with position mismatches"


@shared_task(name="accounting_daily_snapshot", queue="accounting")
@skip_if_not_trading_day("accounting_daily_snapshot")
def accounting_daily_snapshot():
    """
    Genera el EquitySnapshot diario de cada portafolio activo. Las cuentas Alpaca se sincronizan
    antes para que el equity sea el saldo real del broker; sin broker se usa el equity contable.
    """
    from apps.accounting.services import open_positions

    logger.info("Executing accounting_daily_snapshot...")
    today = get_ny_now().date()
    created = 0

    for port in Portfolio.objects.filter(status="active"):
        active_broker = port.broker_accounts.filter(trading_enabled=True).first()
        if active_broker and active_broker.provider == "alpaca":
            sync_alpaca_account(active_broker)

        summary = open_positions(port, refresh=True)["summary"]
        total_eq = summary["equity"]
        positions_val = summary["gross_exposure"]

        EquitySnapshot.objects.update_or_create(
            portfolio=port,
            date=today,
            defaults={
                "total_equity": total_eq,
                "cash_balance": total_eq - positions_val,
                "positions_value": positions_val,
                "source": "broker" if summary["equity_source"] == "broker" else "book",
            }
        )
        created += 1

    return f"Created {created} daily equity snapshots"




@shared_task(name="sync_fundamentals_universe", queue="marketdata")
def sync_fundamentals_universe():
    """Weekly/daily task to sync financial statements for ranked_allocation universe."""
    logger.info("Executing sync_fundamentals_universe...")
    from apps.fundamentals.services import sync_active_universe
    res = sync_active_universe(force=False)
    logger.info(f"Fundamental sync completed: {res['synced']} instruments updated.")
    return res
