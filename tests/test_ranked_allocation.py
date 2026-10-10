import pytest
from decimal import Decimal
import pandas as pd
from django.utils import timezone

from apps.instruments.models import Instrument
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument
from apps.strategies.engines.ranked_allocation import RankedAllocationEngine
from apps.fundamentals.models import FundamentalStatement
from apps.fundamentals.metrics import METRIC_REGISTRY, MetricContext


@pytest.mark.django_db
def test_metrics_registry_and_percentiles():
    """Validates metrics registry definitions and ranking direction."""
    assert "per" in METRIC_REGISTRY
    assert "roic" in METRIC_REGISTRY
    assert "dist_52w_low" in METRIC_REGISTRY

    per_def = METRIC_REGISTRY["per"]
    assert per_def.direction == "asc"  # lower is better

    roic_def = METRIC_REGISTRY["roic"]
    assert roic_def.direction == "desc"  # higher is better


@pytest.mark.django_db
def test_ranked_allocation_engine_hold_mode():
    """Tests ranking multiple assets in 'hold' mode with free rank weights."""
    inst1 = Instrument.objects.create(symbol="AAPL", name="Apple", asset_class="equity")
    inst2 = Instrument.objects.create(symbol="MSFT", name="Microsoft", asset_class="equity")
    inst3 = Instrument.objects.create(symbol="GOOGL", name="Alphabet", asset_class="equity")

    # Add dummy fundamental statements
    # AAPL has high net income
    FundamentalStatement.objects.create(
        instrument=inst1,
        period_type="Q",
        period_end=timezone.now().date(),
        available_at=timezone.now().date(),
        source="fmp",
        data={"revenue": 100000.0, "operating_income": 30000.0, "net_income": 25000.0, "total_equity": 50000.0},
    )
    # MSFT has moderate net income
    FundamentalStatement.objects.create(
        instrument=inst2,
        period_type="Q",
        period_end=timezone.now().date(),
        available_at=timezone.now().date(),
        source="fmp",
        data={"revenue": 80000.0, "operating_income": 20000.0, "net_income": 15000.0, "total_equity": 60000.0},
    )
    # GOOGL missing fundamentals -> should get neutral percentile 0.5

    engine = RankedAllocationEngine()

    dates = pd.date_range("2024-01-01", periods=100)
    ohlcv_data = {
        "AAPL": pd.DataFrame({"Close": [150.0] * 100, "Low": [140.0] * 100, "High": [160.0] * 100}, index=dates),
        "MSFT": pd.DataFrame({"Close": [300.0] * 100, "Low": [280.0] * 100, "High": [320.0] * 100}, index=dates),
        "GOOGL": pd.DataFrame({"Close": [120.0] * 100, "Low": [110.0] * 100, "High": [130.0] * 100}, index=dates),
    }

    params = {
        "metrics": [{"key": "operating_margin", "weight": 1.0}],
        "rank_weights": [0.50, 0.30, 0.20],  # rank 1 gets 50%, rank 2 gets 30%, rank 3 gets 20%
        "execution": {"mode": "hold"},
    }

    targets, diagnostics = engine.compute_signal(ohlcv_data, params, {})

    assert len(targets) == 3
    # Total assigned target exposure must equal 100% (1.0)
    total_exposure = sum(targets.values())
    assert abs(float(total_exposure) - 1.0) < 0.001

    # AAPL (30% op margin) > MSFT (25% op margin) > GOOGL (missing = neutral)
    assert diagnostics["AAPL"]["rank"] == 1
    assert targets["AAPL"] == Decimal("0.50")
    assert diagnostics["GOOGL"]["rank"] in [2, 3]


@pytest.mark.django_db
def test_ranked_allocation_engine_strategy_mode_cash_when_out():
    """In strategy mode, when sub-strategy returns FLAT (0), capital stays in cash."""
    inst1 = Instrument.objects.create(symbol="NVDA", name="Nvidia", asset_class="equity")
    inst2 = Instrument.objects.create(symbol="AMD", name="AMD", asset_class="equity")

    dates = pd.date_range("2024-01-01", periods=50)
    ohlcv_data = {
        "NVDA": pd.DataFrame({"Open": [100.0]*50, "High": [105.0]*50, "Low": [95.0]*50, "Close": [100.0]*50, "Volume": [1000]*50}, index=dates),
        "AMD": pd.DataFrame({"Open": [100.0]*50, "High": [105.0]*50, "Low": [95.0]*50, "Close": [100.0]*50, "Volume": [1000]*50}, index=dates),
    }

    engine = RankedAllocationEngine()
    params = {
        "metrics": [],
        "rank_weights": [0.60, 0.40],
        "execution": {
            "mode": "strategy",
            "strategy_name": "TrendFollowing_GoldCross",
            "params": {"short_ma": 10, "long_ma": 30},
        },
    }

    targets, diagnostics = engine.compute_signal(ohlcv_data, params, {})

    # Flat prices won't trigger Golden Cross -> signals will be 0 -> exposures should be 0.0 (stays in cash)
    for sym, exp in targets.items():
        assert exp == Decimal("0.0")
        assert diagnostics[sym]["execution_mode"] == "strategy"
