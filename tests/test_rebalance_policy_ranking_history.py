import pytest
from datetime import date, datetime, timedelta, timezone as dt_timezone
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
from django.contrib.auth import get_user_model
from django.utils import timezone

from apps.accounting.models import LogicalPosition
from apps.instruments.models import Instrument
from apps.marketdata.models import Bar, LatestQuote
from apps.marketdata.services import MarketDataService, FULL_HISTORY_START
from apps.portfolios.models import Portfolio, Allocation
from apps.strategies.engines.ranked_allocation import RankedAllocationEngine
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument, Signal, AllocationSnapshot
from apps.trading.allocator import PortfolioAllocator
from apps.trading.models import RebalanceRun
from apps.trading.tasks import get_frozen_ranking


# ---------- helpers ----------

def _setup(freq="monthly", tolerance="0.10", held_qty="10", price="110.00", exposure="1.0"):
    """Cartera de 10,000 con un bot al 50% sobre QQQ: target = 5,000 / precio."""
    User = get_user_model()
    user = User.objects.create(email=f"p{User.objects.count()}@test.com", username=f"p{User.objects.count()}")
    portfolio = Portfolio.objects.create(
        user=user, name="Policy", kind="paper", paper_initial_capital=Decimal("10000.00"),
        rebalance_frequency=freq, rebalance_tolerance=Decimal(tolerance),
    )
    qqq = Instrument.objects.create(symbol="QQQ", name="QQQ", asset_class="etf")
    LatestQuote.objects.create(instrument=qqq, price=Decimal(price), timestamp=timezone.now())
    strategy = Strategy.objects.create(name="qqq-bot", slug="qqq-bot", kind="multi_strategy", engine="multi_signal")
    version = StrategyVersion.objects.create(strategy=strategy, version=1, status="live")
    StrategyInstrument.objects.create(version=version, instrument=qqq, role="traded", weight=Decimal("1.0"))
    alloc = Allocation.objects.create(portfolio=portfolio, strategy=strategy, target_weight=Decimal("0.5"))
    Signal.objects.create(version=version, instrument=qqq, as_of=timezone.now(), target_exposure=Decimal(exposure))
    if Decimal(held_qty) > 0:
        LogicalPosition.objects.create(
            portfolio=portfolio, strategy=strategy, instrument=qqq, qty=Decimal(held_qty), avg_cost=Decimal("100.00"),
        )
    return portfolio, alloc, qqq


def _previous_run(portfolio, days_ago=1):
    return RebalanceRun.objects.create(
        portfolio=portfolio, trigger="scheduled", status="done",
        as_of=timezone.now() - timedelta(days=days_ago), equity_base=portfolio.paper_initial_capital,
    )


def _allocate(portfolio, rebalance_date):
    run = RebalanceRun.objects.create(
        portfolio=portfolio, trigger="scheduled", as_of=timezone.now(), equity_base=portfolio.paper_initial_capital,
    )
    with patch("apps.trading.allocator.is_periodic_rebalance_date", return_value=rebalance_date):
        intents = PortfolioAllocator(run).generate_targets_and_intents()
    return run.targets.get(), intents


# ---------- política de rebalanceo del portafolio ----------

@pytest.mark.django_db
def test_open_position_is_not_resized_outside_rebalance_date():
    # Tiene 10 acciones ($1,100) y el target es 45.45 ($5,000): fuera de fecha no se toca
    portfolio, _, _ = _setup(held_qty="10")
    Allocation.objects.update(updated_at=timezone.now() - timedelta(days=30))
    _previous_run(portfolio)

    target, intents = _allocate(portfolio, rebalance_date=False)
    assert target.reason == "hold_until_rebalance_date"
    assert target.delta_qty == Decimal("0")
    assert intents == []


@pytest.mark.django_db
def test_position_is_resized_on_rebalance_date_when_drift_exceeds_tolerance():
    portfolio, _, _ = _setup(held_qty="10")
    Allocation.objects.update(updated_at=timezone.now() - timedelta(days=30))
    _previous_run(portfolio)

    target, intents = _allocate(portfolio, rebalance_date=True)
    assert target.reason == "periodic_monthly_drift_exceeded"
    assert len(intents) == 1
    assert intents[0].net_delta_qty == target.target_qty - Decimal("10")


@pytest.mark.django_db
def test_position_within_tolerance_is_held_on_rebalance_date():
    # 44 acciones = $4,840 contra target de $5,000: deriva 3.2% < 10%
    portfolio, _, _ = _setup(held_qty="44")
    Allocation.objects.update(updated_at=timezone.now() - timedelta(days=30))
    _previous_run(portfolio)

    target, intents = _allocate(portfolio, rebalance_date=True)
    assert target.reason == "hold_within_tolerance"
    assert intents == []


@pytest.mark.django_db
def test_entry_and_exit_trade_any_day_regardless_of_frequency():
    portfolio, _, _ = _setup(freq="never", held_qty="0")
    target, intents = _allocate(portfolio, rebalance_date=False)
    assert target.reason == "signal_entry"
    assert len(intents) == 1

    portfolio2, _, _ = _setup_second_exit_case()
    target, intents = _allocate(portfolio2, rebalance_date=False)
    assert target.reason == "signal_exit"
    assert intents[0].net_delta_qty == Decimal("-10")


def _setup_second_exit_case():
    User = get_user_model()
    user = User.objects.create(email="exit@test.com", username="exit")
    portfolio = Portfolio.objects.create(
        user=user, name="Exit", kind="paper", paper_initial_capital=Decimal("10000.00"), rebalance_frequency="never",
    )
    spy = Instrument.objects.create(symbol="SPY", name="SPY", asset_class="etf")
    LatestQuote.objects.create(instrument=spy, price=Decimal("500.00"), timestamp=timezone.now())
    strategy = Strategy.objects.create(name="spy-bot", slug="spy-bot", kind="multi_strategy", engine="multi_signal")
    version = StrategyVersion.objects.create(strategy=strategy, version=1, status="live")
    StrategyInstrument.objects.create(version=version, instrument=spy, role="traded", weight=Decimal("1.0"))
    alloc = Allocation.objects.create(portfolio=portfolio, strategy=strategy, target_weight=Decimal("0.5"))
    Signal.objects.create(version=version, instrument=spy, as_of=timezone.now(), target_exposure=Decimal("0.0"))
    LogicalPosition.objects.create(
        portfolio=portfolio, strategy=strategy, instrument=spy, qty=Decimal("10"), avg_cost=Decimal("480.00"),
    )
    return portfolio, alloc, spy


@pytest.mark.django_db
def test_allocation_change_since_last_run_forces_resize():
    portfolio, _, _ = _setup(held_qty="10")
    _previous_run(portfolio)  # la asignación se guardó después de este run

    target, intents = _allocate(portfolio, rebalance_date=False)
    assert target.reason == "allocation_change"
    assert len(intents) == 1


@pytest.mark.django_db
def test_rebalance_pending_flag_forces_resize():
    portfolio, _, _ = _setup(held_qty="10")
    Allocation.objects.update(updated_at=timezone.now() - timedelta(days=30))
    _previous_run(portfolio)
    portfolio.rebalance_pending = True
    portfolio.save()

    target, intents = _allocate(portfolio, rebalance_date=False)
    assert target.reason == "manual_rebalance"
    assert len(intents) == 1


@pytest.mark.django_db
def test_small_orders_are_no_longer_filtered_by_min_notional():
    # Cartera con target de $5,000 y entrada de $10 de otro bot: antes se descartaba por < $25
    portfolio, alloc, _ = _setup(held_qty="0")
    alloc.target_weight = Decimal("0.001")  # sleeve de $10
    alloc.save()

    target, intents = _allocate(portfolio, rebalance_date=False)
    assert target.reason == "signal_entry"
    assert len(intents) == 1
    assert intents[0].net_delta_qty * Decimal("110") < Decimal("25")


# ---------- ranking congelado entre fechas de rebalanceo ----------

def _ranked_inputs():
    idx = pd.date_range("2024-01-01", periods=30)
    ohlcv = {
        s: pd.DataFrame({"Close": [100.0] * 30, "High": [101.0] * 30, "Low": [99.0] * 30}, index=idx)
        for s in ("AAA", "BBB", "CCC")
    }
    for s in ohlcv:
        Instrument.objects.create(symbol=s, name=s, asset_class="equity")
    params = {"metrics": [], "rank_weights": [0.5, 0.3, 0.2], "execution": {"mode": "hold"}}
    return ohlcv, params


@pytest.mark.django_db
def test_ranked_engine_reuses_frozen_ranking():
    ohlcv, params = _ranked_inputs()
    engine = RankedAllocationEngine()

    # Sin ranking previo: empate de scores => orden alfabético
    targets, diag = engine.compute_signal(ohlcv, params, {})
    assert targets["AAA"] == Decimal("0.5")
    assert diag["AAA"]["ranking_frozen"] is False

    # Con ranking congelado se respeta ese orden
    targets, diag = engine.compute_signal(ohlcv, {**params, "frozen_ranking": ["CCC", "AAA", "BBB"]}, {})
    assert (targets["CCC"], targets["AAA"], targets["BBB"]) == (Decimal("0.5"), Decimal("0.3"), Decimal("0.2"))
    assert diag["CCC"]["rank"] == 1
    assert diag["CCC"]["ranking_frozen"] is True

    # Si el universo cambió, el ranking congelado se descarta
    targets, diag = engine.compute_signal(ohlcv, {**params, "frozen_ranking": ["CCC", "AAA", "ZZZ"]}, {})
    assert targets["AAA"] == Decimal("0.5")
    assert diag["AAA"]["ranking_frozen"] is False


@pytest.mark.django_db
def test_get_frozen_ranking_only_refreshes_on_portfolio_rebalance_date():
    User = get_user_model()
    user = User.objects.create(email="rank@test.com", username="rank")
    portfolio = Portfolio.objects.create(user=user, name="Ranked", kind="paper", rebalance_frequency="monthly")
    strategy = Strategy.objects.create(name="ranked", slug="ranked", kind="ranked_allocation", engine="ranked_allocation")
    version = StrategyVersion.objects.create(strategy=strategy, version=1, status="live")
    Allocation.objects.create(portfolio=portfolio, strategy=strategy, target_weight=Decimal("0.7"))
    today = date(2026, 10, 13)

    # Sin snapshot previo hay que calcular el ranking
    assert get_frozen_ranking(version, today) is None

    AllocationSnapshot.objects.create(
        version=version, as_of=timezone.now(), weights={},
        diagnostics={"NVDA": {"rank": 2}, "AAPL": {"rank": 1}, "TSM": {"rank": 3}},
    )
    with patch("apps.trading.rebalance_policy.is_last_business_day", return_value=False):
        assert get_frozen_ranking(version, today) == ["AAPL", "NVDA", "TSM"]
    with patch("apps.trading.rebalance_policy.is_last_business_day", return_value=True):
        assert get_frozen_ranking(version, today) is None

    # Un rebalanceo manual pendiente también fuerza recalcular
    portfolio.rebalance_pending = True
    portfolio.save()
    with patch("apps.trading.rebalance_policy.is_last_business_day", return_value=False):
        assert get_frozen_ranking(version, today) is None


# ---------- historial completo ----------

class _FakeProvider:
    def __init__(self):
        self.calls = []

    def get_daily_bars(self, symbol, start_dt):
        self.calls.append(start_dt)
        dates = pd.date_range("2000-01-03", periods=5, freq="B", tz="UTC")
        return pd.DataFrame({
            "Date": dates,
            "open": [1.0, 2.0, None, 4.0, 5.0],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.1, 2.1, 3.1, 4.1, 5.1],
            "adj_close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [100, 200, None, 400, 500],
        })


@pytest.mark.django_db
def test_ensure_history_downloads_full_history_once_then_incremental():
    provider = _FakeProvider()
    svc = MarketDataService(provider=provider)
    inst = Instrument.objects.create(symbol="QQQ", name="QQQ", asset_class="etf")
    assert inst.full_history_synced is False

    assert svc.ensure_history(inst) == 5
    assert provider.calls[0] == FULL_HISTORY_START
    inst.refresh_from_db()
    assert inst.full_history_synced is True
    assert Bar.objects.filter(instrument=inst).count() == 5
    # Los huecos del proveedor se rellenan con el cierre / volumen 0
    gap_bar = Bar.objects.filter(instrument=inst).order_by("date")[2]
    assert gap_bar.open == Decimal("3.1")
    assert gap_bar.volume == Decimal("0")

    # Segunda llamada: solo refresco incremental, sin duplicar barras
    svc.ensure_history(inst)
    assert provider.calls[1] > datetime.now(dt_timezone.utc) - timedelta(days=61)
    assert Bar.objects.filter(instrument=inst).count() == 5
