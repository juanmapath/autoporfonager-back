import numpy as np
import pandas as pd
from typing import Optional, Dict
from apps.strategies.signals.indicators import rolling_slope, get_zscore


def compute_macro_regimes(
    hyg_closes: pd.Series,
    lqd_closes: pd.Series,
    dxy_closes: pd.Series,
    vix_closes: pd.Series,
) -> pd.DataFrame:
    """
    Computes intermarket macro regimes across 8 states:
    - Risk Ratio (HYG/LQD) Z-score slope > 0 -> Risk On (1 else 0)
    - DXY Z-score slope > 0 -> Strong Dollar (1 else 0)
    - VIX > 20 -> High Volatility (1 else 0)
    
    Returns a DataFrame with columns: ['Risk_Ratio', 'Z_Risk', 'Slope_Risk', 'Z_DXY', 'Slope_DXY', 'VIX', 'Regime']
    """
    df = pd.DataFrame(index=hyg_closes.index)
    df["HYG"] = hyg_closes
    df["LQD"] = lqd_closes
    df["DXY"] = dxy_closes
    df["VIX"] = vix_closes
    df = df.ffill().bfill()

    df["Risk_Ratio"] = df["HYG"] / df["LQD"]
    df["Z_Risk"] = get_zscore(df["Risk_Ratio"], window=48)
    df["Z_DXY"] = get_zscore(df["DXY"], window=48)

    df["Slope_Risk"] = rolling_slope(df["Z_Risk"], window=3)
    df["Slope_DXY"] = rolling_slope(df["Z_DXY"], window=3)

    cond_risk_on = np.where(df["Slope_Risk"] > 0, 1, 0)
    cond_dxy_strong = np.where(df["Slope_DXY"] > 0, 1, 0)
    cond_vix_high = np.where(df["VIX"] > 20, 1, 0)

    regimes_list = [
        f"R_{r_on}_{dxy_str}_{vix_hi}"
        for r_on, dxy_str, vix_hi in zip(cond_risk_on, cond_dxy_strong, cond_vix_high)
    ]
    df["Regime"] = regimes_list
    return df
