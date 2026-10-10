import pytest
from decimal import Decimal
from django.utils import timezone

from apps.portfolios.models import Portfolio, Allocation
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument
from apps.instruments.models import Instrument
from apps.accounting.models import LogicalPosition
from apps.trading.models import RebalanceRun
from apps.trading.allocator import PortfolioAllocator


@pytest.mark.django_db
def test_allocator_liquidates_orphan_positions_when_asset_removed():
    """
    If a strategy previously had a position in AAPL, but a new version ONLY trades MSFT,
    the allocator must generate a liquidation order for AAPL (target=0, delta=-current_qty).
    """
    from django.contrib.auth import get_user_model
    User = get_user_model()
    user = User.objects.create(email="test@example.com", username="tester")
    portfolio = Portfolio.objects.create(user=user, name="Test Port", kind="paper", paper_initial_capital=Decimal("100000.00"))
    inst_aapl = Instrument.objects.create(symbol="AAPL", name="Apple", asset_class="equity")
    inst_msft = Instrument.objects.create(symbol="MSFT", name="Microsoft", asset_class="equity")

    strategy = Strategy.objects.create(name="MultiAsset Bot", slug="multiasset-bot", kind="one_strategy")
    version = StrategyVersion.objects.create(strategy=strategy, version=1, status="live")

    # New version ONLY contains MSFT
    StrategyInstrument.objects.create(version=version, instrument=inst_msft, role="traded", weight=Decimal("1.0"))

    # But portfolio currently has an existing position in AAPL from an older version
    LogicalPosition.objects.create(
        portfolio=portfolio,
        strategy=strategy,
        instrument=inst_aapl,
        qty=Decimal("50.0"),
        avg_cost=Decimal("150.00"),
    )

    alloc = Allocation.objects.create(
        portfolio=portfolio,
        strategy=strategy,
        version=version,
        target_weight=Decimal("1.0"),
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

    # Target records check
    targets = run.targets.all()
    aapl_target = targets.filter(instrument=inst_aapl).first()
    assert aapl_target is not None
    assert aapl_target.reason == "liquidate_removed_asset"
    assert aapl_target.target_qty == Decimal("0.0")
    assert aapl_target.current_qty == Decimal("50.0")
    assert aapl_target.delta_qty == Decimal("-50.0")

    # Netting intents should have a sell intent for AAPL of 50 shares
    aapl_intent = [i for i in intents if i.instrument == inst_aapl]
    assert len(aapl_intent) == 1
    assert aapl_intent[0].net_delta_qty == Decimal("-50.0")
