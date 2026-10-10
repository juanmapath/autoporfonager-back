import pytest
import pandas as pd
import numpy as np
from apps.strategies.backtest import BacktestEngine


def test_smart_accumulator_simulation_buy_and_hold():
    """Verify that Smart Accumulator accumulates shares on buy signals and never sells on 0."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Open": [100, 102, 105, 104, 108, 110, 112, 115, 120, 125],
        "High": [101, 103, 106, 105, 109, 111, 113, 116, 121, 126],
        "Low": [99, 101, 104, 103, 107, 109, 111, 114, 119, 124],
        "Close": [100.0, 102.0, 105.0, 104.0, 108.0, 110.0, 112.0, 115.0, 120.0, 125.0],
        "Volume": [1000] * 10,
    }, index=dates)

    # Signal fires on day 0 (buy), then 0 (hold), then day 4 fires buy again
    signals = pd.Series([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], index=dates)

    engine = BacktestEngine(initial_cash=1000.0, commission_per_trade=0.0, slippage_pct=0.0)
    res = engine._simulate_accumulator(
        df=df,
        signals=signals,
        symbol="TEST",
        periodic_deposit=500.0,
        deposit_frequency="weekly",
    )

    assert res["bot_type"] == "smart_accumulator"
    assert res["symbol"] == "TEST"
    assert res["total_invested"] >= 1000.0
    assert res["accumulated_shares"] > 0
    assert res["final_value"] > res["total_invested"]
    assert res["roi_pct"] > 0
    assert len(res["trades"]) >= 1
    # Check that all trades remain open/accumulated
    for t in res["trades"]:
        assert t["open"] is True
        assert t["exit_strategy"] == "Acumulado (Hold)"


def test_backtest_engine_rejects_ranked_and_hold():
    """Verify that backtest engine correctly rejects ranked_allocation and hold bots."""
    engine = BacktestEngine(initial_cash=10000.0)
    
    with pytest.raises(ValueError, match="Ranked Allocation"):
        engine.run_for_config(
            bot_type="ranked_allocation",
            traded_symbol="AAPL",
        )

    with pytest.raises(ValueError, match="Hold & Reserve"):
        engine.run_for_config(
            bot_type="hold",
            traded_symbol="AAPL",
        )
