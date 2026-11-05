"""Métricas de curvas de equity para el estudio y el backtest layered."""
from typing import Dict, Optional

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Caída máxima como fracción negativa (p. ej. -0.35)."""
    if equity is None or len(equity) < 2:
        return 0.0
    return float(drawdown_series(equity).min())


def cagr(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365.25, 1 / 365.25)
    end = float(equity.iloc[-1])
    if end <= 0:
        return -1.0
    return float((end / float(equity.iloc[0])) ** (1 / years) - 1)


def sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else 0.0


def recovery_days(equity: pd.Series) -> Optional[int]:
    """Días calendario de la recuperación más larga (de un máximo hasta volver a superarlo); None si no se recupera."""
    if equity is None or len(equity) < 2:
        return 0
    peak_val, peak_date, longest = equity.iloc[0], equity.index[0], 0
    underwater = False
    for d, v in equity.items():
        if v >= peak_val:
            if underwater:
                longest = max(longest, (d - peak_date).days)
            peak_val, peak_date, underwater = v, d, False
        else:
            underwater = True
    if underwater:
        longest = max(longest, (equity.index[-1] - peak_date).days)
        return None if longest == 0 else -longest  # negativo: sigue bajo el agua
    return longest


def window_drawdown(equity: pd.Series, start, end) -> Optional[float]:
    seg = equity[(equity.index >= pd.Timestamp(start)) & (equity.index <= pd.Timestamp(end))]
    if len(seg) < 2:
        return None
    return max_drawdown(seg)


def summary(equity: pd.Series) -> Dict[str, float]:
    dd = max_drawdown(equity)
    g = cagr(equity)
    return {"cagr": g, "max_dd": dd, "sharpe": sharpe(equity), "calmar": (g / abs(dd)) if dd < 0 else None,
            "recovery_days": recovery_days(equity)}
