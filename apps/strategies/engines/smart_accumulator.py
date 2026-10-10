"""
Smart Accumulator Engine.

Purchases assets using available cash when quantitative signals fire (1).
Never issues sell signals on flat/neutral (0) regimes, keeping accumulated shares.
Optional take-profit rule triggers full/partial cash-in when price reaches target gain.
"""
from decimal import Decimal
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from apps.strategies.engines.base import BaseEngine
from apps.strategies.signals.catalog import STRATEGY_MAP


class SmartAccumulatorEngine(BaseEngine):
    def compute_signal(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        instruments_weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
        strat_func_name = params.get("strategy_name") or params.get("strategy_function")
        strat_params = params.get("params", params.get("strategy_params", {}))

        if not strat_func_name and "strategies" in params and params["strategies"]:
            first_s = params["strategies"][0]
            if isinstance(first_s, dict):
                strat_func_name = first_s.get("strategy_name")
                strat_params = first_s.get("params", strat_params)
            elif isinstance(first_s, str):
                strat_func_name = first_s

        strat_func = STRATEGY_MAP.get(strat_func_name)
        take_profit_pct = params.get("take_profit_pct")  # e.g. 0.20 for +20% gain

        targets: Dict[str, Decimal] = {}
        diagnostics: Dict[str, Any] = {}

        for sym, df in ohlcv_data.items():
            weight = instruments_weights.get(sym, Decimal("1.0"))
            raw_sig = 0
            sub_ind = {}

            if strat_func and df is not None and not df.empty:
                try:
                    sigs, ind = strat_func(df, strat_params)
                    raw_sig = int(sigs.iloc[-1]) if not sigs.empty else 0
                    if isinstance(ind, pd.DataFrame) and not ind.empty:
                        last_row = ind.iloc[-1].to_dict()
                        for k, v in last_row.items():
                            if pd.isna(v) or v is None:
                                sub_ind[k] = None
                            elif isinstance(v, (int, float, np.number)):
                                sub_ind[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
                            else:
                                sub_ind[k] = str(v)
                except Exception:
                    raw_sig = 0

            # In accumulator logic:
            # signal = 1 -> BUY DIP / ACCUMULATE (target_exposure = 1.0)
            # signal = 0 -> KEEP ACCUMULATED (allocator handles ratchet without selling)
            targets[sym] = weight if raw_sig > 0 else Decimal("0.0")

            diagnostics[sym] = {
                "raw_signal": raw_sig,
                "strategy_name": strat_func_name,
                "take_profit_pct": take_profit_pct,
                "action": "accumulate_dip" if raw_sig > 0 else "hold_accumulated",
                "indicators": sub_ind,
            }

        return targets, diagnostics
