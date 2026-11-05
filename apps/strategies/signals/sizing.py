"""
Lógica compartida entre el backtest y los motores en vivo para que ambos
dimensionen las posiciones exactamente igual:
  - conversión de pulsos discretos (1 / -1 / 0) a estado continuo de posición
  - apalancamiento walk-forward según una política (`leverage_policy`): fijo, PF de los trades que
    entraron en el mismo régimen de mercado que hoy (`regime_pf`, la táctica por defecto)
    o capas combinables (`layered`, ver apps/strategies/leverage/)
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

DEFAULT_SLIPPAGE_PCT = 0.0005

PF_MAX_LEVERAGE_THRESHOLD = 2.5
PF_MID_LEVERAGE_THRESHOLD = 1.5


def pulses_to_long_state(signals: pd.Series) -> pd.Series:
    """Convierte pulsos (1: entrada, -1: salida, 0: sin cambio) a estado continuo (True: dentro)."""
    state = np.zeros(len(signals), dtype=bool)
    current = False
    for i, v in enumerate(signals.values):
        if v == 1:
            current = True
        elif v == -1:
            current = False
        state[i] = current
    return pd.Series(state, index=signals.index)


def resolve_leverage_params(params: Dict[str, Any]) -> Tuple[float, float, bool]:
    """Extrae (base_leverage, max_leverage, use_dynamic_leverage) de los params de una StrategyVersion."""
    params = params or {}
    base = float(params.get("leverage") or 1.0)
    max_lev = float(params.get("max_leverage") or base)
    return base, max(base, max_lev), bool(params.get("use_regimes", False))


def profit_factor(pnls: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Ganancias / pérdidas (99 si no hay pérdidas y sí ganancias; 1 si no hay nada). Con pesos, ponderado."""
    if weights is None:
        weights = [1.0] * len(pnls)
    gains = sum(p * w for p, w in zip(pnls, weights) if p > 0)
    losses = sum(abs(p) * w for p, w in zip(pnls, weights) if p <= 0)
    return gains / losses if losses > 0 else (99.0 if gains > 0 else 1.0)


def leverage_from_pf(
    pf: float,
    base_leverage: float,
    max_leverage: float,
    max_threshold: float = PF_MAX_LEVERAGE_THRESHOLD,
    mid_threshold: float = PF_MID_LEVERAGE_THRESHOLD,
) -> float:
    """Mapea un PF a apalancamiento: >= max_threshold -> max; >= mid_threshold -> punto medio; si no, base."""
    if pf >= max_threshold:
        return max_leverage
    if pf >= mid_threshold:
        return base_leverage + 0.5 * (max_leverage - base_leverage)
    return base_leverage


# ---------------------------------------------------------------------------------------------
# Política de apalancamiento (`StrategyVersion.params["leverage_policy"]`)
#   fixed     -> siempre el apalancamiento base
#   regime_pf -> según el PF de los trades cerrados que entraron en el régimen de mercado de hoy
#   layered   -> capas combinables (escaladores, topes, políticas de etiqueta), 04e
# Sin `leverage_policy`: use_regimes=true -> regime_pf; use_regimes=false -> fixed.
# ---------------------------------------------------------------------------------------------

POLICY_FIXED = "fixed"
POLICY_REGIME_PF = "regime_pf"
POLICY_LAYERED = "layered"
POLICY_TYPES = (POLICY_FIXED, POLICY_REGIME_PF, POLICY_LAYERED)
COMBINE_MODES = ("min", "mean")

DEFAULT_MIN_TRADES = 3
DEFAULT_LOOKBACK_YEARS = 15.0


@dataclass(frozen=True)
class LeveragePolicy:
    type: str = POLICY_FIXED
    providers: Tuple[str, ...] = ("macro",)
    combine: str = "min"
    min_trades: int = DEFAULT_MIN_TRADES
    lookback_years: Optional[float] = DEFAULT_LOOKBACK_YEARS
    max_threshold: float = PF_MAX_LEVERAGE_THRESHOLD
    mid_threshold: float = PF_MID_LEVERAGE_THRESHOLD
    provider_params: Mapping[str, Any] = field(default_factory=dict)
    layered: Optional[Any] = None   # LayeredConfig cuando type == "layered"

    @property
    def uses_regimes(self) -> bool:
        if self.type == POLICY_LAYERED:
            return bool(self.layered and self.layered.needs_regimes)
        return self.type == POLICY_REGIME_PF

    @property
    def apply(self) -> str:
        return self.layered.apply if self.type == POLICY_LAYERED and self.layered else "entry"

    def to_dict(self) -> Dict[str, Any]:
        if self.type == POLICY_LAYERED and self.layered:
            return self.layered.to_dict()
        d: Dict[str, Any] = {"type": self.type}
        if self.type == POLICY_REGIME_PF:
            d.update({
                "providers": list(self.providers), "combine": self.combine, "min_trades": self.min_trades,
                "lookback_years": self.lookback_years,
                "thresholds": {"max": self.max_threshold, "mid": self.mid_threshold},
                "provider_params": dict(self.provider_params),
            })
        return d


def resolve_leverage_policy(params: Optional[Mapping[str, Any]]) -> LeveragePolicy:
    """Política efectiva de una versión (ver cabecera de sección)."""
    params = params or {}
    raw = params.get("leverage_policy")
    if not isinstance(raw, Mapping) or not raw.get("type"):
        return LeveragePolicy(type=POLICY_REGIME_PF if params.get("use_regimes") else POLICY_FIXED)

    ptype = str(raw["type"])
    if ptype not in POLICY_TYPES:
        raise ValueError(f"leverage_policy.type invalido: {ptype!r}. Opciones: {', '.join(POLICY_TYPES)}.")
    if ptype == POLICY_LAYERED:
        from apps.strategies.leverage.config import parse_layered

        base = float(params.get("leverage") or 1.0)
        cfg = parse_layered(raw, base, float(params.get("max_leverage") or base))
        return LeveragePolicy(type=ptype, providers=(cfg.provider,), layered=cfg)
    th = raw.get("thresholds") or {}
    combine = str(raw.get("combine") or "min")
    if combine not in COMBINE_MODES:
        raise ValueError(f"leverage_policy.combine invalido: {combine!r}. Opciones: {', '.join(COMBINE_MODES)}.")
    providers = raw.get("providers") or ["macro"]
    lookback = raw.get("lookback_years", DEFAULT_LOOKBACK_YEARS)
    return LeveragePolicy(
        type=ptype,
        providers=tuple(str(p) for p in providers),
        combine=combine,
        min_trades=int(raw.get("min_trades", DEFAULT_MIN_TRADES)),
        lookback_years=float(lookback) if lookback else None,
        max_threshold=float(th.get("max", PF_MAX_LEVERAGE_THRESHOLD)),
        mid_threshold=float(th.get("mid", PF_MID_LEVERAGE_THRESHOLD)),
        provider_params=dict(raw.get("provider_params") or {}),
    )


@dataclass
class ClosedTrade:
    """Trade cerrado para la política: fechas, PnL y régimen (por proveedor) del día de ENTRADA."""
    entry_date: Any
    exit_date: Any
    pnl: float
    regimes: Dict[str, Optional[str]] = field(default_factory=dict)


def _age_years(a, b) -> float:
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25


def regime_pf_estimate(
    policy: LeveragePolicy,
    trades: Sequence[ClosedTrade],
    provider: str,
    label: Optional[str],
    as_of,
) -> Dict[str, Any]:
    """
    PF de los trades cerrados ANTES de `as_of` (dentro de la ventana) que entraron en el régimen `label`.
    `source`: "regime_pf" (hay al menos `min_trades` en el régimen), "base_insufficient_trades"
    (menos trades que el mínimo en este régimen) o "base_no_regime" (hoy no hay etiqueta de régimen).
    Solo en el primer caso `pf_used` es un PF; en los otros el apalancamiento es el base.
    """
    eligible = [t for t in trades if pd.Timestamp(t.exit_date) < pd.Timestamp(as_of)]
    if policy.lookback_years:
        eligible = [t for t in eligible if _age_years(t.exit_date, as_of) <= policy.lookback_years]
    in_regime = [t for t in eligible if label is not None and t.regimes.get(provider) == label]
    n = len(in_regime)
    pf_reg = profit_factor([t.pnl for t in in_regime]) if in_regime else None

    info: Dict[str, Any] = {"regime": label, "n_trades_regime": n, "pf_regime": pf_reg, "pf_used": None}
    if label is None:
        info["source"] = "base_no_regime"
    elif n < policy.min_trades:
        info["source"] = "base_insufficient_trades"
    else:
        info.update(pf_used=pf_reg, source="regime_pf")
    return info


def policy_entry_leverage(
    policy: LeveragePolicy,
    closed_trades: Sequence[ClosedTrade],
    base_leverage: float,
    max_leverage: float,
    as_of=None,
    current_regimes: Optional[Mapping[str, Optional[str]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Apalancamiento de una entrada nueva en `as_of` y diagnóstico. Función ÚNICA para el backtest y
    los motores en vivo. Solo cuentan los trades cerrados antes de `as_of`.
    """
    max_leverage = max(base_leverage, max_leverage)
    diag: Dict[str, Any] = {"leverage_policy": policy.type}

    if policy.type == POLICY_FIXED:
        diag["applied_leverage"] = max(1.0, float(base_leverage))
        return diag["applied_leverage"], diag

    if policy.type == POLICY_LAYERED:
        raise ValueError("La politica 'layered' se evalua en el simulador (apps.strategies.leverage.simulator).")

    current_regimes = current_regimes or {}
    suggestions: List[float] = []
    per_provider: Dict[str, Any] = {}
    for provider in policy.providers:
        info = regime_pf_estimate(policy, closed_trades, provider, current_regimes.get(provider), as_of)
        pf_used = info["pf_used"]
        suggestions.append(
            base_leverage if pf_used is None
            else leverage_from_pf(pf_used, base_leverage, max_leverage, policy.max_threshold, policy.mid_threshold)
        )
        per_provider[provider] = info
    if not suggestions:
        combined = base_leverage
    elif policy.combine == "mean":
        combined = sum(suggestions) / len(suggestions)
    else:
        combined = min(suggestions)
    lev = max(1.0, float(combined))
    diag.update({"providers": per_provider, "combine": policy.combine, "applied_leverage": lev})
    return lev, diag


def aligned_label(labels: Optional[Sequence[Any]], i: int) -> Optional[str]:
    if labels is None or i >= len(labels) or i < 0:
        return None
    v = labels[i]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v)


def simulate_trade_leverages(
    closes: Sequence[float],
    long_state: Sequence[Any],
    policy: LeveragePolicy,
    base_leverage: float,
    max_leverage: float,
    dates: Optional[Sequence[Any]] = None,
    regime_labels: Optional[Mapping[str, Sequence[Any]]] = None,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
) -> Tuple[float, Dict[str, Any]]:
    """
    Reproduce la simulación del backtest sobre el histórico y devuelve (apalancamiento fijado en la
    entrada del trade abierto o el de la próxima entrada, diagnóstico).
    `regime_labels[proveedor][i]` es la etiqueta de régimen alineada con `closes[i]`.
    """
    max_leverage = max(base_leverage, max_leverage)
    regime_labels = regime_labels or {}
    cash, shares, entry_price = 1.0, 0.0, 0.0
    in_position = False
    trade_leverage = base_leverage
    trade_diag: Dict[str, Any] = {}
    closed: List[ClosedTrade] = []
    entry_idx = 0
    entry_regimes: Dict[str, Optional[str]] = {}

    def day(i):
        return dates[i] if dates is not None else i

    for i, (price, is_long) in enumerate(zip(closes, long_state)):
        price = float(price)
        if is_long and not in_position:
            regimes_now = {p: aligned_label(regime_labels.get(p), i) for p in policy.providers}
            active, diag = policy_entry_leverage(policy, closed, base_leverage, max_leverage, day(i), regimes_now)
            adj_price = price * (1.0 + slippage_pct)
            investable = cash * active
            if investable > 0 and adj_price > 0:
                shares = investable / adj_price
                cash -= shares * adj_price
                entry_price = adj_price
                trade_leverage, trade_diag = active, diag
                entry_idx, entry_regimes = i, regimes_now
                in_position = True
        elif not is_long and in_position:
            adj_price = price * (1.0 - slippage_pct)
            proceeds = shares * adj_price
            cash += proceeds
            closed.append(ClosedTrade(day(entry_idx), day(i), proceeds - (shares * entry_price), entry_regimes))
            shares = 0.0
            in_position = False

    if in_position:
        return float(trade_leverage), trade_diag
    n = len(closes)
    regimes_next = {p: aligned_label(regime_labels.get(p), n - 1) for p in policy.providers}
    next_day = (pd.Timestamp(day(n - 1)) + pd.Timedelta(days=1)) if dates is not None and n else n
    return policy_entry_leverage(policy, closed, base_leverage, max_leverage, next_day, regimes_next)


def current_trade_leverage(
    closes: Sequence[float],
    long_state: Sequence[Any],
    base_leverage: float = 1.0,
    max_leverage: float = 1.0,
    use_dynamic_leverage: bool = False,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    policy: Optional[LeveragePolicy] = None,
    dates: Optional[Sequence[Any]] = None,
    regime_labels: Optional[Mapping[str, Sequence[Any]]] = None,
    return_info: bool = False,
    layer_inputs: Optional[Any] = None,
):
    """
    Reproduce la simulación de BacktestEngine._simulate sobre el histórico y devuelve el
    apalancamiento fijado en la entrada del trade actualmente abierto. El valor se decide
    al entrar y no cambia mientras el trade siga abierto, igual que en el backtest.
    Si no hay posición abierta devuelve el apalancamiento que aplicaría la próxima entrada.
    Sin `policy` se usa la legada (`use_dynamic_leverage` -> regime_pf, si no fixed).
    """
    if policy is None:
        policy = LeveragePolicy(type=POLICY_REGIME_PF if use_dynamic_leverage else POLICY_FIXED)
    if policy.type == POLICY_LAYERED:
        from apps.strategies.leverage.simulator import live_layered_leverage

        lev, info = live_layered_leverage(
            closes, long_state, dates, policy, layer_inputs, regime_labels, slippage_pct=slippage_pct,
        )
        return (lev, info) if return_info else lev
    lev, info = simulate_trade_leverages(
        closes, long_state, policy, base_leverage, max_leverage, dates, regime_labels, slippage_pct
    )
    return (lev, info) if return_info else lev
