"""
Catálogo de capas de apalancamiento (04e §3). Cada capa es causal: su valor en la fecha t solo usa
datos hasta el cierre de t.

Tipos (`kind`):
  - scaler: multiplicador continuo m >= 0 (entra multiplicando).
  - cap:    apalancamiento máximo permitido hoy (entra con min()). Un tope "apagado" devuelve L_max.
  - label:  política de etiqueta: apalancamiento absoluto desde etiqueta de régimen + trades pasados;
            se evalúa dentro del simulador porque depende de los trades cerrados.

Las capas `path=True` (drawdown_brake) dependen de la curva de equity del propio bot y también se
evalúan dentro del simulador.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class LayerSpec:
    key: str
    kind: str                       # scaler | cap | label
    name: str
    description: str
    defaults: Dict[str, Any]
    symbols: Callable[[Dict[str, Any]], List[str]] = lambda p: []
    compute: Optional[Callable[["LayerInputs", Dict[str, Any], pd.DatetimeIndex, float], pd.Series]] = None
    path: bool = False              # depende de la equity del bot (se evalúa en el simulador)
    needs_regimes: bool = False     # usa etiquetas de régimen (proveedor macro)
    references: str = ""
    experimental: bool = False


@dataclass
class LayerInputs:
    """Series de cierre por símbolo (DatetimeIndex) que necesitan las capas; `self` es el activo operado."""
    closes: Dict[str, pd.Series] = field(default_factory=dict)

    def series(self, symbol: str) -> Optional[pd.Series]:
        s = self.closes.get(symbol)
        if s is None or len(s) == 0:
            return None
        return s


def _ref_symbol(params: Dict[str, Any]) -> str:
    return str(params.get("ref") or "self")


def _on(index, s: Optional[pd.Series]) -> Optional[pd.Series]:
    """Último valor conocido de `s` en cada fecha de `index` (causal)."""
    if s is None:
        return None
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    union = s.index.union(index)
    return s.reindex(union).ffill().reindex(index)


# ------------------------------- escaladores -------------------------------

def _vol_target(inputs: LayerInputs, p: Dict[str, Any], index, lmax: float) -> pd.Series:
    close = inputs.series(_ref_symbol(p))
    if close is None:
        return pd.Series(1.0, index=index)
    rets = close.pct_change()
    if p.get("estimator", "ewma") == "max20_60":
        sigma = pd.concat([rets.rolling(20).std(), rets.rolling(60).std()], axis=1).max(axis=1)
    else:
        lam = float(p.get("lambda", 0.94))
        var = (rets ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=20).mean()
        sigma = np.sqrt(var)
    sigma_ann = sigma * np.sqrt(TRADING_DAYS)
    m = float(p.get("target_vol", 0.25)) / sigma_ann
    m = m.replace([np.inf, -np.inf], np.nan)
    return _on(index, m).fillna(1.0)


# ------------------------------- topes -------------------------------

def _hysteresis(above: pd.Series, below: pd.Series, initial: bool = True) -> pd.Series:
    """Estado binario con banda: se enciende cuando `above`, se apaga cuando `below`; si no, mantiene."""
    state = np.empty(len(above), dtype=bool)
    cur = initial
    for i, (a, b) in enumerate(zip(above.values, below.values)):
        if a:
            cur = True
        elif b:
            cur = False
        state[i] = cur
    return pd.Series(state, index=above.index)


def _trend_gate(inputs: LayerInputs, p: Dict[str, Any], index, lmax: float) -> pd.Series:
    close = inputs.series(_ref_symbol(p))
    if close is None:
        return pd.Series(lmax, index=index)
    n, band = int(p.get("sma", 200)), float(p.get("band", 0.01))
    sma = close.rolling(n).mean()
    on = _hysteresis((close > sma * (1 + band)).fillna(False), (close < sma * (1 - band)).fillna(False), True)
    on = on.where(sma.notna(), True)  # sin SMA todavía: sin restricción
    cap = pd.Series(np.where(on, lmax, float(p.get("cap_below", 1.0))), index=close.index)
    return _on(index, cap).fillna(lmax)


def _binary_cap(stress: pd.Series, index, lmax: float, cap: float) -> pd.Series:
    vals = pd.Series(np.where(stress.fillna(False).astype(bool), cap, lmax), index=stress.index)
    return _on(index, vals).fillna(lmax)


def _vix_term(inputs: LayerInputs, p: Dict[str, Any], index, lmax: float) -> pd.Series:
    vix, vix3m = inputs.series("^VIX"), inputs.series("^VIX3M")
    if vix is None or vix3m is None:
        return pd.Series(lmax, index=index)
    df = pd.concat([vix.rename("v"), vix3m.rename("v3")], axis=1).ffill()
    stress = (df["v"] / df["v3"]) >= float(p.get("threshold", 1.0))
    stress = stress.where(df["v3"].notna(), False)
    return _binary_cap(stress, index, lmax, float(p.get("cap", 1.0)))


def _vix_relative(inputs: LayerInputs, p: Dict[str, Any], index, lmax: float) -> pd.Series:
    vix = inputs.series("^VIX")
    if vix is None:
        return pd.Series(lmax, index=index)
    med = vix.rolling(int(p.get("window", 252)), min_periods=60).median()
    stress = (vix / med) >= float(p.get("ratio", 1.3))
    return _binary_cap(stress.where(med.notna(), False), index, lmax, float(p.get("cap", 1.0)))


def _credit_stress(inputs: LayerInputs, p: Dict[str, Any], index, lmax: float) -> pd.Series:
    hyg, lqd = inputs.series("HYG"), inputs.series("LQD")
    if hyg is None or lqd is None:
        return pd.Series(lmax, index=index)
    ratio = pd.concat([hyg.rename("h"), lqd.rename("l")], axis=1).ffill().pipe(lambda d: d["h"] / d["l"])
    sma = ratio.rolling(int(p.get("sma", 50))).mean()
    chg = ratio.pct_change(int(p.get("change_window", 20)))
    stress = (ratio < sma) & (chg < float(p.get("drop", -0.01)))
    return _binary_cap(stress.where(sma.notna() & chg.notna(), False), index, lmax, float(p.get("cap", 1.0)))


TURBULENCE_BASKET = ["SPY", "TLT", "GLD", "HYG", "UUP"]


def _turbulence(inputs: LayerInputs, p: Dict[str, Any], index, lmax: float) -> pd.Series:
    basket = p.get("basket") or TURBULENCE_BASKET
    series = [inputs.series(s) for s in basket]
    if any(s is None for s in series):
        return pd.Series(lmax, index=index)
    prices = pd.concat([s.rename(sym) for s, sym in zip(series, basket)], axis=1).ffill()
    rets = prices.pct_change().dropna()
    window, pct = int(p.get("window", 252)), float(p.get("p", 0.90))
    vals = rets.values
    dist = np.full(len(rets), np.nan)
    for i in range(window, len(rets)):
        hist = vals[i - window:i]  # ventana ANTERIOR a t (sin el dia t)
        mu = hist.mean(axis=0)
        cov = np.cov(hist, rowvar=False)
        try:
            inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            continue
        d = vals[i] - mu
        dist[i] = float(d @ inv @ d)
    dist = pd.Series(dist, index=rets.index)
    threshold = dist.rolling(window, min_periods=60).quantile(pct).shift(1)  # umbral con historia previa
    stress = (dist > threshold).where(threshold.notna(), False)
    return _binary_cap(stress, index, lmax, float(p.get("cap", 1.0)))


LAYERS: Dict[str, LayerSpec] = {}


def _register(spec: LayerSpec):
    LAYERS[spec.key] = spec


_register(LayerSpec(
    "vol_target", "scaler", "Volatilidad objetivo",
    "Escala el apalancamiento a volatilidad objetivo / volatilidad estimada del activo (EWMA anualizada): "
    "menos apalancamiento cuando la volatilidad sube.",
    {"target_vol": 0.25, "estimator": "ewma", "lambda": 0.94, "ref": "self"},
    symbols=lambda p: [] if _ref_symbol(p) == "self" else [_ref_symbol(p)],
    compute=_vol_target, references="Moreira–Muir 2017; Harvey et al. 2018",
))
_register(LayerSpec(
    "trend_gate", "cap", "Filtro de tendencia",
    "Sobre la media móvil de n días se permite el máximo; debajo, el tope `cap_below`. Con banda para no "
    "cambiar de estado a diario.",
    {"sma": 200, "band": 0.01, "cap_below": 1.0, "ref": "self"},
    symbols=lambda p: [] if _ref_symbol(p) == "self" else [_ref_symbol(p)],
    compute=_trend_gate, references="Faber 2007; Gayed–Bilello 2016",
))
_register(LayerSpec(
    "vix_term", "cap", "Estrés VIX (curva invertida)",
    "Si VIX / VIX3M >= umbral (curva de volatilidad invertida) el tope es `cap`.",
    {"threshold": 1.0, "cap": 1.0}, symbols=lambda p: ["^VIX", "^VIX3M"], compute=_vix_term,
    references="Estrés de volatilidad implícita",
))
_register(LayerSpec(
    "vix_relative", "cap", "VIX relativo",
    "Si VIX / mediana de 252 días >= `ratio` el tope es `cap`.",
    {"ratio": 1.3, "window": 252, "cap": 1.0}, symbols=lambda p: ["^VIX"], compute=_vix_relative,
))
_register(LayerSpec(
    "credit_stress", "cap", "Estrés de crédito",
    "Si HYG/LQD está bajo su media de n días y cayó más de `drop` en 20 días, el tope es `cap`.",
    {"sma": 50, "drop": -0.01, "change_window": 20, "cap": 1.0}, symbols=lambda p: ["HYG", "LQD"],
    compute=_credit_stress, references="Gilchrist–Zakrajšek 2012",
))
_register(LayerSpec(
    "turbulence", "cap", "Turbulencia",
    "Distancia de Mahalanobis de los retornos de una canasta (SPY, TLT, GLD, HYG, UUP) frente a su año "
    "previo; sobre el percentil `p` el tope es `cap`.",
    {"window": 252, "p": 0.90, "cap": 1.0}, symbols=lambda p: list(p.get("basket") or TURBULENCE_BASKET),
    compute=_turbulence, references="Kritzman–Li 2010; Kritzman–Page–Turkington 2012",
))
_register(LayerSpec(
    "drawdown_brake", "cap", "Freno por caída",
    "Tope proporcional a la distancia al límite de caída de la curva del bot: "
    "L_max × (1 − caída actual / límite), con mínimo L_min.",
    {"dd_limit": 0.30}, path=True, references="Grossman–Zhou 1993",
))
_register(LayerSpec(
    "macro_pf", "label", "PF por régimen macro",
    "El PF de los trades pasados que entraron en el mismo régimen macro que hoy (regime_pf de 04d): "
    "PF >= 2.5 máximo, >= 1.5 punto medio, si no base.",
    {"min_trades": 3, "lookback_years": 15}, needs_regimes=True, references="Sistema actual (04d)",
))
_register(LayerSpec(
    "kelly_regime", "label", "Kelly por régimen",
    "Kelly fraccional con los trades del régimen actual: fraction × media / varianza de sus retornos.",
    {"fraction": 0.5, "min_trades": 3, "lookback_years": 15}, needs_regimes=True, references="Kelly; Thorp",
))


def layer_catalog() -> List[Dict[str, Any]]:
    return [
        {"key": s.key, "kind": s.kind, "name": s.name, "description": s.description, "defaults": dict(s.defaults),
         "needs_regimes": s.needs_regimes, "path_dependent": s.path, "references": s.references,
         "experimental": s.experimental}
        for s in LAYERS.values()
    ]
