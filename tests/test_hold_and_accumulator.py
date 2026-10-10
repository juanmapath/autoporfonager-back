import pytest
from decimal import Decimal
import pandas as pd
from django.utils import timezone
from django.contrib.auth import get_user_model

from apps.portfolios.models import Portfolio, Allocation
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument
from apps.instruments.models import Instrument
from apps.accounting.models import LogicalPosition
from apps.trading.models import RebalanceRun
from apps.trading.allocator import PortfolioAllocator
from apps.strategies.engines.hold import HoldEngine
from apps.strategies.engines.smart_accumulator import SmartAccumulatorEngine


@pytest.mark.django_db
def test_hold_engine_and_cash_reserve():
    engine = HoldEngine()
    
    # 1. Asset Hold
    targets, diag = engine.compute_signal({}, {"is_cash": False}, {"SPY": Decimal("1.0")})
    assert targets["SPY"] == Decimal("1.0")
    assert diag["SPY"]["is_cash"] is False

    # 2. Cash Reserve Hold
    targets_cash, diag_cash = engine.compute_signal({}, {"is_cash": True}, {"CASH": Decimal("1.0")})
    assert targets_cash["CASH"] == Decimal("0.0")
    assert diag_cash["CASH"]["is_cash"] is True


@pytest.mark.django_db
def test_hold_bot_frozen_in_allocator_when_not_participating():
    User = get_user_model()
    user = User.objects.create(email="hold@test.com", username="holder")
    portfolio = Portfolio.objects.create(user=user, name="Hold Port", kind="paper", paper_initial_capital=Decimal("100000.00"))
    inst = Instrument.objects.create(symbol="GLD", name="Gold", asset_class="etf")

    strategy = Strategy.objects.create(name="Gold Hold", slug="gold-hold", kind="hold", engine="hold")
    version = StrategyVersion.objects.create(
        strategy=strategy,
        version=1,
        params={"participates_in_rebalance": False},  # FROZEN
        status="live",
    )
    StrategyInstrument.objects.create(version=version, instrument=inst, role="traded", weight=Decimal("1.0"))

    # Currently holding 100 shares of GLD
    LogicalPosition.objects.create(
        portfolio=portfolio,
        strategy=strategy,
        instrument=inst,
        qty=Decimal("100.0"),
        avg_cost=Decimal("180.00"),
    )

    Allocation.objects.create(
        portfolio=portfolio,
        strategy=strategy,
        version=version,
        target_weight=Decimal("0.50"),
        enabled=True,
    )

    run = RebalanceRun.objects.create(
        portfolio=portfolio,
        trigger="scheduled",
        as_of=timezone.now(),
        equity_base=Decimal("100000.00"),
    )

    allocator = PortfolioAllocator(run)
    intents = allocator.generate_targets_and_intents()

    # GLD target must equal current_shares (100) -> delta = 0 -> no trade intents
    target = run.targets.filter(instrument=inst).first()
    assert target is not None
    assert target.target_qty == Decimal("100.0")
    assert target.delta_qty == Decimal("0.0")
    assert len([i for i in intents if i.instrument == inst]) == 0


@pytest.mark.django_db
def test_smart_accumulator_no_sell_on_flat():
    User = get_user_model()
    user = User.objects.create(email="acc@test.com", username="accumulator")
    portfolio = Portfolio.objects.create(user=user, name="Acc Port", kind="paper", paper_initial_capital=Decimal("100000.00"))
    inst = Instrument.objects.create(symbol="BTC", name="Bitcoin", asset_class="crypto")

    strategy = Strategy.objects.create(name="BTC Accumulator", slug="btc-acc", kind="smart_accumulator", engine="smart_accumulator")
    version = StrategyVersion.objects.create(strategy=strategy, version=1, status="live")
    StrategyInstrument.objects.create(version=version, instrument=inst, role="traded", weight=Decimal("1.0"))

    # Already accumulated 50 shares
    LogicalPosition.objects.create(
        portfolio=portfolio,
        strategy=strategy,
        instrument=inst,
        qty=Decimal("50.0"),
        avg_cost=Decimal("30000.00"),
    )

    Allocation.objects.create(
        portfolio=portfolio,
        strategy=strategy,
        version=version,
        target_weight=Decimal("0.50"),
        enabled=True,
    )

    run = RebalanceRun.objects.create(
        portfolio=portfolio,
        trigger="scheduled",
        as_of=timezone.now(),
        equity_base=Decimal("100000.00"),
    )

    # When signal exposure is 0 (FLAT): target must retain current 50 shares without selling
    allocator = PortfolioAllocator(run)
    intents = allocator.generate_targets_and_intents()

    target = run.targets.filter(instrument=inst).first()
    assert target is not None
    assert target.target_qty == Decimal("50.0")
    assert target.delta_qty == Decimal("0.0")
