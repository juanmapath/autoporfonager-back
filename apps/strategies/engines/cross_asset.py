from typing import Dict, Any, Tuple
from decimal import Decimal
import pandas as pd
from apps.strategies.engines.base import BaseEngine
from apps.strategies.signals.catalog import STRATEGY_MAP
from apps.strategies.signals.sizing import pulses_to_long_state, resolve_leverage_params, current_trade_leverage


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

        # Pulsos (1 / -1 / 0) -> estado continuo de posición, igual que en el backtest
        long_state = pulses_to_long_state(signals)
        is_long = bool(long_state.iloc[-1]) if not long_state.empty else False
        base_leverage, max_leverage, use_dynamic_leverage = resolve_leverage_params(params)

        targets = {}
        leverages = {}
        for sym, weight in instruments_weights.items():
            traded_df = ohlcv_data.get(sym)
            applied_leverage = 1.0
            if is_long and traded_df is not None and not traded_df.empty:
                traded_state = long_state.reindex(traded_df.index).fillna(False).astype(bool)
                applied_leverage = current_trade_leverage(
                    traded_df["Close"].values, traded_state.values, base_leverage, max_leverage, use_dynamic_leverage
                )
            leverages[sym] = round(applied_leverage, 4)
            targets[sym] = weight * Decimal(str(round(applied_leverage, 4))) if is_long else Decimal("0.0")

        diagnostics = {
            "source_symbol": source_symbol,
            "raw_signal": last_sig,
            "is_long": is_long,
            "applied_leverage": leverages,
            "targets": {k: str(v) for k, v in targets.items()},
        }
        return targets, diagnostics
