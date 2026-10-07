from typing import Dict, Any, Tuple, List
from decimal import Decimal
import numpy as np
import pandas as pd
from apps.strategies.engines.base import BaseEngine
from apps.strategies.signals.catalog import STRATEGY_MAP


class MultiSignalEngine(BaseEngine):
    """
    Computes signals combining multiple sub-strategies using logical OR / continuous state tracking,
    identically to trade_model/xmain.py.
    """

    def compute_signal(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        instruments_weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
        strategies: List[str] = params.get("strategies", [])
        strategies_params: List[Any] = params.get("strategies_params", [])

        targets = {}
        diagnostics = {}

        for symbol, df in ohlcv_data.items():
            if df is None or df.empty or not strategies:
                targets[symbol] = Decimal("0.0")
                continue

            all_signals = []
            all_indicators = []
            all_signal_types = []

            for s_name, s_param in zip(strategies, strategies_params):
                strat_func = STRATEGY_MAP.get(s_name)
                if not strat_func:
                    continue
                sig, ind = strat_func(df, s_param)
                s_type = ind["SignalType"] if "SignalType" in ind.columns else pd.Series("default", index=df.index)
                all_signals.append(sig)
                all_indicators.append(ind)
                all_signal_types.append(s_type)

            if not all_signals:
                targets[symbol] = Decimal("0.0")
                continue

            # State tracking: convert each strategy to active long state (matching xmain.py lines 188-206)
            active_states = []
            for sig in all_signals:
                state = pd.Series(False, index=sig.index)
                current_state = False
                for i in range(len(sig)):
                    val = sig.iloc[i]
                    if val == 1:
                        current_state = True
                    elif val == -1:
                        current_state = False
                    state.iloc[i] = current_state
                active_states.append(state)

            combined_long = pd.Series(False, index=df.index)
            for state in active_states:
                combined_long = combined_long | state

            is_long = combined_long.iloc[-1]
            weight = instruments_weights.get(symbol, Decimal("1.0"))
            exposure = weight if is_long else Decimal("0.0")
            targets[symbol] = exposure

            active_names = [strategies[i] for i, s in enumerate(active_states) if s.iloc[-1]]
            diagnostics[symbol] = {
                "active_strategies": active_names,
                "is_long": bool(is_long),
                "exposure": str(exposure),
            }

        return targets, diagnostics
