from typing import Dict, Any, Tuple
from decimal import Decimal
import pandas as pd
from apps.strategies.engines.base import BaseEngine
from apps.strategies.signals.catalog import STRATEGY_MAP


class SingleSignalEngine(BaseEngine):
    """
    Computes signals using a single technical or statistical strategy function
    from the centralized catalog.
    """

    def compute_signal(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        instruments_weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
        strategy_func_name = params.get("strategy_name") or params.get("strategy_function")

        strategy_params = params.get("strategy_params", params.get("params", {}))

        # Soporte para bots donde la estrategia viene en params["strategies"] = [{"strategy_name": ..., "params": ...}]
        if not strategy_func_name and "strategies" in params and params["strategies"]:
            first_s = params["strategies"][0]
            if isinstance(first_s, dict):
                strategy_func_name = first_s.get("strategy_name")
                strategy_params = first_s.get("params", strategy_params)
            elif isinstance(first_s, str):
                strategy_func_name = first_s

        strat_func = STRATEGY_MAP.get(strategy_func_name)
        if not strat_func:
            raise ValueError(f"Strategy function '{strategy_func_name}' not found in centralized catalog.")

        targets = {}
        diagnostics = {}

        for symbol, df in ohlcv_data.items():
            if df is None or df.empty:
                targets[symbol] = Decimal("0.0")
                continue

            signals, indicators = strat_func(df, strategy_params)
            last_signal = int(signals.iloc[-1]) if not signals.empty else 0
            
            # Map signal to exposure: 1 -> long (scaled by weight), 0/-1 -> flat (0)
            weight = instruments_weights.get(symbol, Decimal("1.0"))
            exposure = weight if last_signal > 0 else Decimal("0.0")
            targets[symbol] = exposure

            # Store last indicator diagnostics
            last_ind = {}
            if isinstance(indicators, pd.DataFrame) and not indicators.empty:
                last_row = indicators.iloc[-1].to_dict()
                last_ind = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in last_row.items()}
            diagnostics[symbol] = {
                "raw_signal": last_signal,
                "exposure": str(exposure),
                "indicators": last_ind,
            }

        return targets, diagnostics
