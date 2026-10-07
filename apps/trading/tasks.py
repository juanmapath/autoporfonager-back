import logging
from datetime import date
from decimal import Decimal
from celery import shared_task
from django.utils import timezone

from apps.instruments.models import Instrument
from apps.marketdata.services import MarketDataService
from apps.marketdata.models import Bar, LatestQuote
from apps.strategies.models import Strategy, StrategyVersion, Signal
from apps.strategies.engines import get_engine
from apps.portfolios.models import Portfolio
from apps.trading.models import RebalanceRun
from apps.trading.allocator import PortfolioAllocator
from apps.trading.execution import OrderDispatcher
from apps.accounting.models import LogicalPosition, EquitySnapshot
from core.market_time import is_trading_day, get_ny_now

logger = logging.getLogger(__name__)


@shared_task(name="marketdata_refresh", queue="marketdata")
def marketdata_refresh():
    """Descarga barras incrementales y LatestQuote de todos los instrumentos activos."""
    logger.info("Executing marketdata_refresh...")
    svc = MarketDataService()
    instruments = Instrument.objects.filter(tradable=True)
    for inst in instruments:
        svc.sync_bars(inst.symbol, days=30)
        svc.update_latest_quote(inst.symbol)
    return f"Refreshed {instruments.count()} instruments"


@shared_task(name="compute_regimes_and_signals", queue="signals")
def compute_regimes_and_signals():
    """Calcula régimen macro y corre compute_signals para cada versión de estrategia activa."""
    logger.info("Executing compute_regimes_and_signals...")
    versions = StrategyVersion.objects.filter(status="live", strategy__is_active=True).select_related("strategy")
    signals_computed = 0

    for v in versions:
        strat = v.strategy
        engine = get_engine(strat.engine)

        instruments = v.instruments.all().select_related("instrument")
        ohlcv_dict = {}
        weights = {}

        for si in instruments:
            sym = si.instrument.symbol
            weights[sym] = si.weight
            bars = Bar.objects.filter(instrument=si.instrument).order_by("date")
            if bars.exists():
                import pandas as pd
                df = pd.DataFrame([{
                    "Date": b.date,
                    "Open": float(b.open),
                    "High": float(b.high),
                    "Low": float(b.low),
                    "Close": float(b.close),
                    "Volume": float(b.volume),
                } for b in bars]).set_index("Date")
                ohlcv_dict[sym] = df

        if not ohlcv_dict:
            continue

        try:
            targets, diagnostics = engine.compute_signal(ohlcv_dict, v.params, weights)
            for sym, exposure in targets.items():
                inst = Instrument.objects.filter(symbol=sym).first()
                if inst:
                    last_sig = Signal.objects.filter(version=v, instrument=inst).order_by("-as_of").first()
                    changed = (last_sig.target_exposure != exposure) if last_sig else True
                    Signal.objects.create(
                        version=v,
                        instrument=inst,
                        as_of=timezone.now(),
                        target_exposure=exposure,
                        changed=changed,
                        diagnostics=diagnostics.get(sym, {}),
                    )
                    signals_computed += 1
        except Exception as e:
            logger.error(f"Error computing signal for {strat.slug} v{v.version}: {e}")

    return f"Computed {signals_computed} signals"


@shared_task(name="orchestrate_portfolio_runs", queue="execution")
def orchestrate_portfolio_runs():
    """Evalúa rebalanceo y genera targets e intents para portafolios activos."""
    logger.info("Executing orchestrate_portfolio_runs...")
    portfolios = Portfolio.objects.filter(status="active", trading_enabled=True)
    created_runs = 0
    today_ny = get_ny_now().date()

    for p in portfolios:
        # Idempotencia: evitar corridas duplicadas en el mismo día para el mismo portafolio
        already_run = RebalanceRun.objects.filter(
            portfolio=p,
            as_of__date=today_ny,
            status__in=["planned", "executing", "done"]
        ).exists()
        if already_run:
            logger.info(f"Portfolio '{p.name}' (id={p.id}) ya fue rebalanceado hoy ({today_ny}). Omitiendo.")
            continue

        run = RebalanceRun.objects.create(
            portfolio=p,
            trigger="scheduled",
            as_of=timezone.now(),
            status="planned",
            equity_base=p.paper_initial_capital if p.kind == "paper" else Decimal("100000.00"),
        )
        allocator = PortfolioAllocator(run)
        intents = allocator.generate_targets_and_intents()
        if intents:
            run.status = "executing"
            run.save(update_fields=["status"])
        else:
            run.status = "done"
            run.save(update_fields=["status"])
        created_runs += 1

    return f"Orchestrated {created_runs} portfolio runs"


@shared_task(name="dispatch_moc_and_manual_orders", queue="execution")
def dispatch_moc_and_manual_orders():
    """Despacha órdenes MOC a Alpaca y genera los fills simulados + avisos de Telegram."""
    logger.info("Executing dispatch_moc_and_manual_orders...")
    active_runs = RebalanceRun.objects.filter(status="executing")
    dispatcher = OrderDispatcher()
    dispatched_count = 0

    for run in active_runs:
        for intent in run.intents.filter(status="pending"):
            try:
                dispatcher.dispatch_intent(intent)
                dispatched_count += 1
            except Exception as e:
                logger.error(f"Failed to dispatch intent {intent.id}: {e}")
        run.status = "done"
        run.save(update_fields=["status"])

    return f"Dispatched {dispatched_count} orders"


@shared_task(name="run_daily_trading_pipeline", queue="execution")
def run_daily_trading_pipeline():
    """
    Cadena inmediata (15:50 NY):
    1. Descarga cotizaciones y barras (marketdata_refresh)
    2. Calcula regímenes y señales cuantitativas (compute_regimes_and_signals)
    3. Rebalancea portafolios generando intenciones de orden (orchestrate_portfolio_runs)
    4. Despacha inmediatamente las órdenes MOC/Market a Alpaca o simulador (dispatch_moc_and_manual_orders)
    """
    logger.info(">>> [PIPELINE 15:50 NY] Iniciando pipeline automatizado consecutivo...")
    try:
        r1 = marketdata_refresh()
        logger.info(f"[1/4] Market data refresh: {r1}")
    except Exception as e:
        logger.error(f"[1/4] Market data refresh falló: {e}")

    try:
        r2 = compute_regimes_and_signals()
        logger.info(f"[2/4] Regimes & signals: {r2}")
    except Exception as e:
        logger.error(f"[2/4] Compute signals falló: {e}")

    try:
        r3 = orchestrate_portfolio_runs()
        logger.info(f"[3/4] Portfolio rebalance: {r3}")
    except Exception as e:
        logger.error(f"[3/4] Portfolio rebalance falló: {e}")

    try:
        r4 = dispatch_moc_and_manual_orders()
        logger.info(f"[4/4] Order dispatch: {r4}")
    except Exception as e:
        logger.error(f"[4/4] Order dispatch falló: {e}")

    return "Daily trading pipeline completed successfully."


@shared_task(name="reconcile_broker_fills", queue="execution")
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
    ).exclude(broker_order_id="").select_related("portfolio", "broker_account", "instrument")

    dispatcher = OrderDispatcher()
    reconciled_fills = 0
    synced_accounts = set()

    for order in pending_orders:
        account = order.broker_account
        if not account or not account.credential:
            continue

        creds = account.credential.get_secrets()
        api_key = creds.get("key_id")
        api_secret = creds.get("secret")
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
                        portfolio=order.portfolio,
                        broker_account=account,
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


@shared_task(name="accounting_daily_snapshot", queue="accounting")
def accounting_daily_snapshot():
    """Generates daily EquitySnapshot and balances for all active portfolios."""
    logger.info("Executing accounting_daily_snapshot...")
    today = date.today()
    created = 0

    for port in Portfolio.objects.filter(status="active"):
        positions_val = Decimal("0.00")
        for pos in port.logical_positions.filter(qty__gt=0):
            quote = LatestQuote.objects.filter(instrument=pos.instrument).first()
            px = quote.price if quote else pos.avg_cost
            positions_val += pos.qty * px

        active_broker = port.broker_accounts.filter(trading_enabled=True).first()
        if active_broker and active_broker.managed_capital:
            total_eq = active_broker.managed_capital
        else:
            total_eq = port.paper_initial_capital if port.kind == "paper" else Decimal("100000.00")

        EquitySnapshot.objects.update_or_create(
            portfolio=port,
            date=today,
            defaults={
                "total_equity": total_eq,
                "cash_balance": max(Decimal("0.00"), total_eq - positions_val),
                "positions_value": positions_val,
                "source": "daily_scheduler",
            }
        )
        created += 1

    return f"Created {created} daily equity snapshots"

