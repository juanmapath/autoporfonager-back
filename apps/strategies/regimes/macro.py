"""Proveedor macro: crédito (HYG/LQD), dólar (DXY) y volatilidad (VIX) -> 8 regímenes."""
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from apps.strategies.signals.indicators import get_zscore, rolling_slope

DEFAULTS = {"zscore_window": 48, "slope_window": 3, "vix_threshold": 20.0}


def _close(bars: Dict[str, pd.DataFrame], symbol: str) -> pd.Series:
    df = bars.get(symbol)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df["Close"].astype(float).copy()
    s.index = pd.to_datetime(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()


def macro_regime_frame(
    hyg: pd.Series, lqd: pd.Series, dxy: pd.Series, vix: pd.Series,
    zscore_window: int = 48, slope_window: int = 3, vix_threshold: float = 20.0,
) -> pd.DataFrame:
    """
    - HYG/LQD: pendiente del z-score > 0 -> risk on (1)
    - DXY: pendiente del z-score > 0 -> dólar fuerte (1)
    - VIX > umbral -> volatilidad alta (1)
    Regime = R_{risk_on}_{dxy_fuerte}_{vix_alto}. Causal: solo se rellena hacia adelante (ffill); las
    filas sin datos suficientes quedan con Regime None (no se rellena hacia atrás).
    """
    idx = hyg.index.union(lqd.index).union(dxy.index).union(vix.index).sort_values()
    df = pd.DataFrame({
        "HYG": hyg.reindex(idx), "LQD": lqd.reindex(idx), "DXY": dxy.reindex(idx), "VIX": vix.reindex(idx),
    }).ffill()

    df["Risk_Ratio"] = df["HYG"] / df["LQD"]
    df["Z_Risk"] = get_zscore(df["Risk_Ratio"], window=zscore_window)
    df["Z_DXY"] = get_zscore(df["DXY"], window=zscore_window)
    df["Slope_Risk"] = rolling_slope(df["Z_Risk"], window=slope_window)
    df["Slope_DXY"] = rolling_slope(df["Z_DXY"], window=slope_window)

    valid = df[["Slope_Risk", "Slope_DXY", "VIX"]].notna().all(axis=1)
    risk_on = (df["Slope_Risk"] > 0).astype(int)
    dxy_strong = (df["Slope_DXY"] > 0).astype(int)
    vix_high = (df["VIX"] > vix_threshold).astype(int)
    labels = "R_" + risk_on.astype(str) + "_" + dxy_strong.astype(str) + "_" + vix_high.astype(str)
    df["Regime"] = labels.where(valid, None)
    return df


class MacroRegimeProvider:
    key = "macro"
    SYMBOLS = ["HYG", "LQD", "DX-Y.NYB", "^VIX"]

    def required_symbols(self) -> List[str]:
        return list(self.SYMBOLS)

    def _frame(self, bars: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> pd.DataFrame:
        p = {**DEFAULTS, **(params or {})}
        return macro_regime_frame(
            _close(bars, "HYG"), _close(bars, "LQD"), _close(bars, "DX-Y.NYB"), _close(bars, "^VIX"),
            zscore_window=int(p["zscore_window"]), slope_window=int(p["slope_window"]),
            vix_threshold=float(p["vix_threshold"]),
        )

    def compute(self, bars: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> pd.Series:
        return self._frame(bars, params)["Regime"]

    def features(self, bars: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, Any]:
        """Indicadores de la última fecha (para RegimeSnapshot.features)."""
        df = self._frame(bars, params)
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {
            k: (None if pd.isna(row[k]) else float(row[k]))
            for k in ("Risk_Ratio", "Z_Risk", "Slope_Risk", "Z_DXY", "Slope_DXY", "VIX")
        }
