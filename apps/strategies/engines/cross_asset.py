from typing import Dict, Any, Tuple
from decimal import Decimal
import pandas as pd
from apps.strategies.engines.base import BaseEngine
from apps.strategies.signals.catalog import STRATEGY_MAP


class CrossAssetEngine(BaseEngine):
    """
    Computes signals on a source asset (e.g. QQQ) and applies the resulting exposure
    to one or more traded instruments (e.g. TQQQ, QLD).
    """

    def compute_signal(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        instruments_weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
        source_symbol = params.get("source_symbol")
        strat_func_name = params.get("strategy_name")
        strat_params = params.get("params", {})

        if not strat_func_name and "strategies" in params and params["strategies"]:
            first_s = params["strategies"][0]
            if isinstance(first_s, dict):
                strat_func_name = first_s.get("strategy_name")
                strat_params = first_s.get("params", strat_params)
            elif isinstance(first_s, str):
                strat_func_name = first_s

        strat_func = STRATEGY_MAP.get(strat_func_name)
        if not strat_func:
            raise ValueError(f"Strategy '{strat_func_name}' not found.")


        source_df = ohlcv_data.get(source_symbol)
        if source_df is None or source_df.empty:
            return {sym: Decimal("0.0") for sym in instruments_weights}, {"error": "Missing source asset data"}

        signals, ind = strat_func(source_df, strat_params)

        last_sig = int(signals.iloc[-1]) if not signals.empty else 0

        targets = {}
        for sym, weight in instruments_weights.items():
            targets[sym] = weight if last_sig > 0 else Decimal("0.0")

        diagnostics = {
            "source_symbol": source_symbol,
            "raw_signal": last_sig,
            "targets": {k: str(v) for k, v in targets.items()},
        }
        return targets, diagnostics


class FollowPriceEngine(BaseEngine):
    """
    Follow Price Engine with trailing stop and price thresholds.
    """

    def compute_signal(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        instruments_weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
        targets = {}
        diagnostics = {}
        trailing_pct = Decimal(str(params.get("trailing_pct", 0.05)))

        for sym, df in ohlcv_data.items():
            if df is None or df.empty:
                targets[sym] = Decimal("0.0")
                continue
            close = df["Close"].iloc[-1]
            high_recent = df["High"].rolling(window=20).max().iloc[-1]
            drawdown = (high_recent - close) / high_recent if high_recent > 0 else 0
            is_above_stop = drawdown < float(trailing_pct)
            weight = instruments_weights.get(sym, Decimal("1.0"))
            targets[sym] = weight if is_above_stop else Decimal("0.0")
            diagnostics[sym] = {"drawdown_from_high": float(drawdown), "active": is_above_stop}

        return targets, diagnostics
