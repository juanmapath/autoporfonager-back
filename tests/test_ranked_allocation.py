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


@pytest.mark.django_db
def test_ranked_allocation_per_asset_regime_leverage_hold_mode():
    """Tests that each asset in RankedAllocationEngine gets individual dynamic leverage based on its historical regime."""
    Instrument.objects.create(symbol="STRONG", name="Strong Uptrend", asset_class="equity")
    Instrument.objects.create(symbol="WEAK", name="Weak Downtrend", asset_class="equity")

    dates = pd.date_range("2024-01-01", periods=60)
    # STRONG has steady positive daily gains -> high Gain-to-Pain ratio and above mean -> bull_strong -> max_leverage (2.0x)
    strong_closes = [100.0 + i * 1.5 for i in range(60)]
    # WEAK has steady daily losses -> below mean -> neutral_or_weak -> base_leverage (1.0x)
    weak_closes = [100.0 - i * 0.5 for i in range(60)]

    ohlcv_data = {
        "STRONG": pd.DataFrame({"Close": strong_closes, "Low": strong_closes, "High": strong_closes}, index=dates),
        "WEAK": pd.DataFrame({"Close": weak_closes, "Low": weak_closes, "High": weak_closes}, index=dates),
    }

    engine = RankedAllocationEngine()
    params = {
        "metrics": [],
        "rank_weights": [0.60, 0.40],
        "leverage": 1.0,
        "max_leverage": 2.0,
        "use_regimes": True,
        "execution": {"mode": "hold"},
    }

    targets, diagnostics = engine.compute_signal(ohlcv_data, params, {})

    # Alphabetical tie-break: STRONG is rank 1 (assigned_weight = 0.60)
    # Plus strong bull regime -> applied_leverage = 2.0 -> target_exposure = 0.60 * 2.0 = 1.20
    assert diagnostics["STRONG"]["rank"] == 1
    assert diagnostics["STRONG"]["regime_state"] == "bull_strong"
    assert diagnostics["STRONG"]["applied_leverage"] == 2.0
    assert targets["STRONG"] == Decimal("1.20")

    # WEAK has rank 2 (assigned_weight = 0.40) and weak regime -> applied_leverage = 1.0 -> target_exposure = 0.40
    assert diagnostics["WEAK"]["rank"] == 2
    assert diagnostics["WEAK"]["regime_state"] == "neutral_or_weak"
    assert diagnostics["WEAK"]["applied_leverage"] == 1.0
    assert targets["WEAK"] == Decimal("0.40")




# ---------------------------------------------------------------------------
# F1: missing data ranks last
# ---------------------------------------------------------------------------
from apps.fundamentals.metrics import MetricDef

NAN = float("nan")
INF = float("inf")


def _pcts(values, direction="desc"):
    syms = list(values.keys())
    return RankedAllocationEngine._metric_percentiles(values, syms, direction)


def test_pct_all_valid_matches_legacy_formula():
    vals = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}
    pct = _pcts(vals, "desc")
    assert [pct[s] for s in "ABCDE"] == [1.0, 0.75, 0.5, 0.25, 0.0]
    pct_asc = _pcts(vals, "asc")
    assert [pct_asc[s] for s in "EDCBA"] == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_pct_two_missing_go_last():
    vals = {"A": 10.0, "B": 20.0, "C": 30.0, "D": None, "E": None}
    pct = _pcts(vals, "asc")  # lower is better
    assert (pct["A"], pct["B"], pct["C"]) == (1.0, 0.75, 0.5)
    assert pct["D"] == 0.0 and pct["E"] == 0.0


@pytest.mark.parametrize("direction", ["asc", "desc"])
def test_pct_missing_last_both_directions(direction):
    vals = {"A": 1.0, "B": 2.0, "C": None}
    pct = _pcts(vals, direction)
    assert pct["C"] == 0.0
    assert min(pct["A"], pct["B"]) > pct["C"]


def test_pct_single_valid_of_four():
    pct = _pcts({"A": 1.0, "B": None, "C": None, "D": None})
    assert pct == {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}


def test_pct_no_valid_all_neutral():
    assert set(_pcts({"A": None, "B": NAN, "C": INF}).values()) == {0.5}


def test_pct_single_asset_neutral():
    assert _pcts({"A": 3.0}) == {"A": 0.5}


def test_pct_inf_and_nan_treated_as_missing():
    pct = _pcts({"A": 1.0, "B": INF, "C": -INF, "D": NAN, "E": 2.0}, "desc")
    assert pct["E"] == 1.0 and pct["A"] == 0.75
    assert pct["B"] == pct["C"] == pct["D"] == 0.0


def _patch_metrics(monkeypatch, data):
    """data: {metric_key: {symbol: value}} -> registers fake metrics (desc)."""
    reg = {}
    for key, per_sym in data.items():
        reg[key] = MetricDef(
            key=key, label=key, direction="desc", source="fundamental", description="",
            fn=(lambda ctx, per_sym=per_sym: per_sym.get(ctx.instrument.symbol)),
        )
    monkeypatch.setattr("apps.strategies.engines.ranked_allocation.METRIC_REGISTRY", reg)


def _ohlcv(symbols):
    dates = pd.date_range("2024-01-01", periods=20)
    return {s: pd.DataFrame({"Close": [10.0] * 20}, index=dates) for s in symbols}


@pytest.mark.django_db
def test_engine_missing_metric_lowers_rank_and_reports_it(monkeypatch):
    for s in ("AAA", "BBB", "CCC"):
        Instrument.objects.create(symbol=s, name=s, asset_class="equity")
    params = {
        "metrics": [{"key": "m1", "weight": 1.0}, {"key": "m2", "weight": 0.5}],
        "rank_weights": [0.5, 0.3, 0.2],
        "execution": {"mode": "hold"},
    }
    engine = RankedAllocationEngine()
    m2 = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0}

    # Baseline: BBB has data in m1 and ranks 2nd.
    _patch_metrics(monkeypatch, {"m1": {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0}, "m2": m2})
    _, base = engine.compute_signal(_ohlcv(["AAA", "BBB", "CCC"]), params, {})
    assert base["BBB"]["rank"] == 2
    assert base["BBB"]["missing_metrics"] == []

    # BBB without m1 -> last place in m1 -> drops below CCC.
    _patch_metrics(monkeypatch, {"m1": {"AAA": 3.0, "BBB": None, "CCC": 1.0}, "m2": m2})
    _, diag = engine.compute_signal(_ohlcv(["AAA", "BBB", "CCC"]), params, {})
    assert diag["BBB"]["rank"] == 3
    assert diag["BBB"]["percentiles"]["m1"] == 0.0
    assert diag["BBB"]["missing_metrics"] == ["m1"]
    assert diag["AAA"]["missing_metrics"] == []
