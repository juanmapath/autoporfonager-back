import numpy as np
import pandas as pd
from typing import Dict, Optional


def generate_signals_multi_state(
    entry_conditions: Dict[str, pd.Series],
    exit_conditions: Dict[str, pd.Series],
    valid_mask: pd.Series,
    cooldowns: Optional[Dict[str, int]] = None,
    max_days: Optional[Dict[str, int]] = None,
):
    """
    Generic state-machine utility that generates chronological trading signals (1, -1, 0)
    and signal types (long, short, etc.).
    
    entry_conditions: dict {state_name: pd.Series(bool)}
    exit_conditions: dict {state_name: pd.Series(bool)}
    valid_mask: pd.Series(bool) indicator warmup mask
    cooldowns: optional dict {state_name: int} minimum bars out before re-entering
    max_days: optional dict {state_name: int} maximum bars in position
    """
    keys = list(entry_conditions.keys())
    n = len(valid_mask)
    if cooldowns is None:
        cooldowns = {}
    if max_days is None:
        max_days = {}

    signals = np.zeros(n)
    sig_types = ["none"] * n
    active_state = "none"
    bars_out = {k: 9999 for k in keys}
    bars_in = 0

    entry_arrs = {k: entry_conditions[k].values for k in keys}
    exit_arrs = {k: exit_conditions[k].values for k in keys}
    valid_arr = valid_mask.values

    for i in range(n):
        if not valid_arr[i]:
            signals[i] = 0
            sig_types[i] = "none"
            active_state = "none"
            bars_in = 0
            continue

        if active_state == "none":
            for k in keys:
                bars_out[k] += 1
        else:
            bars_in += 1
            for k in keys:
                if k != active_state:
                    bars_out[k] += 1

        # 1. Evaluate exit
        if active_state != "none":
            max_d = max_days.get(active_state, 999999)
            time_exit = bars_in >= max_d

            if exit_arrs[active_state][i] or time_exit:
                signals[i] = -1
                bars_out[active_state] = 0
                active_state = "none"
                bars_in = 0
            elif active_state == "meanreversion" and "trendfollowing" in entry_arrs and entry_arrs["trendfollowing"][i]:
                signals[i] = 0
                active_state = "trendfollowing"
                bars_out["trendfollowing"] = 0
                bars_in = 0
            else:
                signals[i] = 0

        # 2. Evaluate entry
        if active_state == "none":
            for k in keys:
                min_cd = cooldowns.get(k, 0)
                if entry_arrs[k][i] and bars_out[k] >= min_cd:
                    signals[i] = 1
                    active_state = k
                    bars_out[k] = 0
                    bars_in = 0
                    break

        sig_types[i] = active_state

    final_signals = pd.Series(signals, index=valid_mask.index)
    signal_type = pd.Series(sig_types, index=valid_mask.index)
    return final_signals, signal_type
