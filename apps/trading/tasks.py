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

    for p in portfolios:
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
