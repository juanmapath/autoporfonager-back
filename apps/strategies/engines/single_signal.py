from typing import Dict, Any, Tuple
from decimal import Decimal
import math
import numpy as np
import pandas as pd
from apps.strategies.engines.base import BaseEngine
from apps.strategies.signals.catalog import STRATEGY_MAP
from apps.strategies.signals.sizing import pulses_to_long_state, resolve_leverage_params, current_trade_leverage


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

        base_leverage, max_leverage, use_dynamic_leverage = resolve_leverage_params(params)

        targets = {}
        diagnostics = {}

        for symbol, df in ohlcv_data.items():
            if df is None or df.empty:
                targets[symbol] = Decimal("0.0")
                continue

            signals, indicators = strat_func(df, strategy_params)
            last_signal = int(signals.iloc[-1]) if not signals.empty else 0

            # Las estrategias emiten pulsos (1: entrada, -1: salida, 0: sin cambio): la posición se
            # mantiene entre la entrada y la salida, igual que en el backtest.
            long_state = pulses_to_long_state(signals)
            is_long = bool(long_state.iloc[-1]) if not long_state.empty else False

            # Apalancamiento fijado en la entrada del trade abierto (misma lógica que el backtest)
            applied_leverage = 1.0
            if is_long:
                applied_leverage = current_trade_leverage(
                    df["Close"].values, long_state.values, base_leverage, max_leverage, use_dynamic_leverage
                )

            weight = instruments_weights.get(symbol, Decimal("1.0"))
            exposure = weight * Decimal(str(round(applied_leverage, 4))) if is_long else Decimal("0.0")
            targets[symbol] = exposure

            # Store last indicator diagnostics (sanitize NaNs/Infs for valid PostgreSQL JSON)
            last_ind = {}
            if isinstance(indicators, pd.DataFrame) and not indicators.empty:
                last_row = indicators.iloc[-1].to_dict()
                for k, v in last_row.items():
                    if pd.isna(v) or v is None:
                        last_ind[k] = None
                    elif isinstance(v, (int, float, np.number)):
                        last_ind[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
                    else:
                        last_ind[k] = str(v)

            diagnostics[symbol] = {
                "raw_signal": last_signal,
                "is_long": is_long,
                "applied_leverage": round(applied_leverage, 4),
                "exposure": str(exposure),
                "indicators": last_ind,
            }

        return targets, diagnostics
