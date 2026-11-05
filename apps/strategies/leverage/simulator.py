"""
Simulador de apalancamiento unificado (04e §2 y §6): una sola función para la política `layered`
en vivo y en backtest (paridad) y para el estudio `leverage_study`, que también corre aquí las
referencias fixed / global_pf / regime_pf con los mismos costes.

Por día t (precio de cierre p_t):
  1. Interés de margen sobre la caja negativa del día anterior (tasa ^IRX + spread) y, opcional,
     rendimiento de la caja positiva.
  2. Llamada de margen: si equity / valor de la posición < mantenimiento, liquidación forzada; la
     configuración queda marcada y no vuelve a entrar hasta una NUEVA señal de entrada.
  3. Señal: entrada (apalancamiento objetivo de hoy), salida, o -con apply="daily"- redimensionado
     si |ΔL| >= min_change (bajadas inmediatas, subidas como mucho max_step_up cada step_days).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from apps.strategies.leverage.config import LayeredConfig, LayerRef
from apps.strategies.leverage.layers import LayerInputs
from apps.strategies.signals.sizing import (
    DEFAULT_SLIPPAGE_PCT, POLICY_LAYERED, POLICY_REGIME_PF, ClosedTrade, LeveragePolicy, aligned_label,
    policy_entry_leverage,
)

DEFAULT_MARGIN_SPREAD = 0.02       # sobre ^IRX; poner la tasa real del broker
DEFAULT_MAINTENANCE = 0.30
RATE_SYMBOL = "^IRX"


@dataclass
class SimCosts:
    slippage: float = DEFAULT_SLIPPAGE_PCT
    commission: float = 0.0
    margin_spread: float = DEFAULT_MARGIN_SPREAD
    maintenance: float = DEFAULT_MAINTENANCE
    cash_yield: bool = False
    margin_interest: bool = True


@dataclass
class SimResult:
    dates: pd.DatetimeIndex
    equity: np.ndarray
    leverage: np.ndarray            # exposición / equity al cierre (0 fuera de mercado)
    applied: np.ndarray             # apalancamiento objetivo vigente (0 fuera de mercado)
    trades: List[Dict[str, Any]]
    margin_calls: List[str]
    interest_paid: float
    resizes: int
    turnover: float                 # suma |Δ valor negociado| / equity
    in_position: bool
    applied_leverage: float         # L vigente al final (o el de una entrada hoy)
    last_info: Dict[str, Any] = field(default_factory=dict)
    resized_last_day: bool = False


def _rates(inputs: Optional[LayerInputs], index, spread: float) -> np.ndarray:
    """Tasa anual (decimal) por día: ^IRX/100 + spread; sin datos, solo el spread."""
    s = inputs.series(RATE_SYMBOL) if inputs else None
    if s is None:
        return np.full(len(index), spread)
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    aligned = s.reindex(s.index.union(index)).ffill().reindex(index).fillna(0.0)
    return aligned.values / 100.0 + spread


class _LayeredEvaluator:
    """Evalúa L objetivo de la política layered para el día i (escaladores × etiquetas, topes, clip)."""

    def __init__(self, cfg: LayeredConfig, index, inputs: LayerInputs, regime_labels: Mapping[str, Sequence]):
        self.cfg = cfg
        self.index = index
        self.labels = regime_labels or {}
        self.static_scalers: Dict[str, np.ndarray] = {}
        self.static_caps: Dict[str, np.ndarray] = {}
        self.path_caps: List[LayerRef] = []
        for ref in cfg.scalers:
            self.static_scalers[ref.key] = ref.spec.compute(inputs, ref.resolved_params(), index, cfg.max).values
        for ref in cfg.caps:
            if ref.spec.path:
                self.path_caps.append(ref)
            else:
                self.static_caps[ref.key] = ref.spec.compute(inputs, ref.resolved_params(), index, cfg.max).values
        self._memo: Dict[Tuple, float] = {}

    # --- políticas de etiqueta (sobre retornos SIN apalancar de los trades del propio bot) ---
    def _label_leverage(self, ref: LayerRef, trades: List[ClosedTrade], i: int) -> Tuple[float, Dict[str, Any]]:
        cfg, p = self.cfg, ref.resolved_params()
        label = aligned_label(self.labels.get(cfg.provider), i)
        key = (ref.key, len(trades), label, i if p.get("lookback_years") else None)
        as_of = self.index[i]
        if ref.key == "macro_pf":
            pol = LeveragePolicy(type=POLICY_REGIME_PF, providers=(cfg.provider,),
                                 min_trades=int(p.get("min_trades", 3)),
                                 lookback_years=float(p["lookback_years"]) if p.get("lookback_years") else None)
            lev, diag = policy_entry_leverage(pol, trades, cfg.base, cfg.max, as_of, {cfg.provider: label})
            info = diag["providers"][cfg.provider]
            return lev, {"regime": label, "n_trades_regime": info["n_trades_regime"], "pf_regime": info["pf_regime"],
                         "source": info["source"]}
        # kelly_regime
        lookback = p.get("lookback_years")
        rets = [t.pnl for t in trades
                if pd.Timestamp(t.exit_date) < pd.Timestamp(as_of) and label is not None
                and t.regimes.get(cfg.provider) == label
                and (not lookback or (pd.Timestamp(as_of) - pd.Timestamp(t.exit_date)).days / 365.25 <= float(lookback))]
        n = len(rets)
        info = {"regime": label, "n_trades_regime": n, "kelly": None,
                "source": "base_no_regime" if label is None else "base_insufficient_trades"}
        if label is None or n < int(p.get("min_trades", 3)):
            return cfg.base, info
        mu, var = float(np.mean(rets)), float(np.var(rets, ddof=1)) if n > 1 else 0.0
        f = float(p.get("fraction", 0.5)) * (mu / var) if var > 0 else (cfg.max if mu > 0 else cfg.min)
        info.update(kelly=f, source="kelly_regime")
        return float(np.clip(f, cfg.min, cfg.max)), info

    def target(self, i: int, trades: List[ClosedTrade], equity: float, peak: float, detail: bool = False):
        cfg = self.cfg
        m = 1.0
        scalers = {}
        for k, arr in self.static_scalers.items():
            v = float(arr[i]) if np.isfinite(arr[i]) else 1.0
            scalers[k] = v
            m *= max(0.0, v)
        labels = {}
        if cfg.label_policies:
            vals = []
            for ref in cfg.label_policies:
                memo_key = (ref.key, len(trades), aligned_label(self.labels.get(cfg.provider), i))
                if not detail and memo_key in self._memo and not ref.resolved_params().get("lookback_years"):
                    lv = self._memo[memo_key]
                    info = {}
                else:
                    lv, info = self._label_leverage(ref, trades, i)
                    self._memo[memo_key] = lv
                vals.append(lv)
                labels[ref.key] = {"leverage": lv, **info}
            lab = min(vals) if cfg.combine_labels == "min" else sum(vals) / len(vals)
            m *= lab / cfg.base
        l_raw = cfg.base * m
        caps = {k: float(arr[i]) for k, arr in self.static_caps.items()}
        dd = 1.0 - equity / peak if peak > 0 else 0.0
        for ref in self.path_caps:  # drawdown_brake
            limit = float(ref.resolved_params().get("dd_limit", 0.30))
            caps[ref.key] = max(cfg.min, cfg.max * (1.0 - dd / limit)) if limit > 0 else cfg.max
        capped = min([l_raw] + list(caps.values()))
        lev = float(np.clip(capped, cfg.min, cfg.max))
        if not detail:
            return lev, None
        cap_min = min(caps.values()) if caps else None
        if capped < cfg.min - 1e-12:
            active = "L_min"
        elif cap_min is not None and cap_min < l_raw - 1e-12 and cap_min <= cfg.max + 1e-12:
            active = min(caps, key=caps.get)
        elif l_raw > cfg.max + 1e-12:
            active = "L_max"
        else:
            active = "scalers" if (scalers or labels) else "base"
        return lev, {"scalers": scalers, "caps": caps, "labels": labels, "L_raw": l_raw, "L": lev,
                     "active_layer": active, "drawdown": float(dd)}


def simulate(
    closes: Sequence[float],
    long_state: Sequence[Any],
    dates,
    policy: LeveragePolicy,
    base_leverage: float = 1.0,
    max_leverage: float = 1.0,
    inputs: Optional[LayerInputs] = None,
    regime_labels: Optional[Mapping[str, Sequence]] = None,
    costs: Optional[SimCosts] = None,
    initial_cash: float = 1.0,
    peak_reset_date=None,
) -> SimResult:
    costs = costs or SimCosts()
    index = pd.to_datetime(pd.Index(dates))
    closes = np.asarray(closes, dtype=float)
    states = [bool(x) for x in long_state]
    n = len(closes)
    layered = policy.type == POLICY_LAYERED
    cfg: Optional[LayeredConfig] = policy.layered if layered else None
    if layered:
        base_leverage, max_leverage = cfg.base, cfg.max
    max_leverage = max(base_leverage, max_leverage)
    # Copia: las entradas se comparten entre bots; "self" es el activo operado de ESTA simulación
    inputs = LayerInputs(closes={"self": pd.Series(closes, index=index), **(inputs.closes if inputs else {})})
    # (si el llamador pasa "self" con más historia -p. ej. el calentamiento del backtest- se usa esa)
    regime_labels = regime_labels or {}
    evaluator = _LayeredEvaluator(cfg, index, inputs, regime_labels) if layered else None
    rates = _rates(inputs, index, costs.margin_spread)
    daily = layered and cfg.apply == "daily"

    cash, shares = float(initial_cash), 0.0
    equity_arr, lev_arr, applied_arr = np.zeros(n), np.zeros(n), np.zeros(n)
    trades: List[Dict[str, Any]] = []
    closed_pnl: List[ClosedTrade] = []       # PnL en $ (global_pf / regime_pf)
    closed_ret: List[ClosedTrade] = []       # retornos sin apalancar (políticas de etiqueta de layered)
    margin_calls: List[str] = []
    interest_paid, turnover, resizes = 0.0, 0.0, 0
    in_pos, blocked = False, False
    applied = base_leverage
    entry_idx, entry_price, entry_equity = 0, 0.0, 0.0
    entry_regimes: Dict[str, Optional[str]] = {}
    last_up_idx = -10 ** 9
    peak = float(initial_cash)
    last_info: Dict[str, Any] = {}
    last_resize = -1
    lev_sum, lev_days = 0.0, 0
    # (estudio) el freno por caída mide la caída desde el inicio del periodo evaluado, no desde la historia previa
    reset_idx = int(np.searchsorted(index.values, np.datetime64(pd.Timestamp(peak_reset_date)))) if peak_reset_date is not None else -1

    def regimes_at(i):
        return {p: aligned_label(regime_labels.get(p), i) for p in policy.providers}

    def target_at(i, equity, detail=False):
        if layered:
            return evaluator.target(i, closed_ret, equity, peak, detail)
        lev, diag = policy_entry_leverage(policy, closed_pnl, base_leverage, max_leverage, index[i], regimes_at(i))
        return lev, diag

    def close_trade(i, price, forced=False):
        nonlocal cash, shares, in_pos, turnover
        adj = price * (1.0 - costs.slippage)
        proceeds = shares * adj - costs.commission
        turnover += abs(shares * adj)
        cash += proceeds
        eq_exit = cash
        ret = adj / entry_price - 1.0 if entry_price > 0 else 0.0
        pnl = eq_exit - entry_equity
        trades.append({
            "entry_date": str(index[entry_idx].date()), "exit_date": str(index[i].date()),
            "entry_price": round(entry_price, 4), "exit_price": round(adj, 4), "return_pct": round(ret * 100, 3),
            "pnl": round(pnl, 6), "leverage_entry": round(entry_lev, 4),
            "avg_leverage": round(lev_sum / lev_days, 4) if lev_days else round(entry_lev, 4),
            "forced": forced, "regimes": dict(entry_regimes),
        })
        closed_pnl.append(ClosedTrade(index[entry_idx], index[i], round(pnl, 2) if initial_cash >= 100 else pnl,
                                      entry_regimes))
        closed_ret.append(ClosedTrade(index[entry_idx], index[i], ret, entry_regimes))
        shares = 0.0
        in_pos = False

    entry_lev = base_leverage
    for i in range(n):
        price = closes[i]
        # 1. intereses / rendimiento de la caja
        if i > 0:
            daily_rate = rates[i - 1] / 252.0
            if cash < 0 and costs.margin_interest:
                interest = -cash * daily_rate
                cash -= interest
                interest_paid += interest
            elif cash > 0 and costs.cash_yield:
                cash += cash * max(0.0, rates[i - 1] - costs.margin_spread) / 252.0  # T-bill, sin spread
        equity = cash + shares * price
        if i == reset_idx:
            peak = equity
        peak = max(peak, equity)

        # 2. llamada de margen
        if in_pos and shares > 0:
            pos_val = shares * price
            if pos_val > 0 and equity / pos_val < costs.maintenance:
                margin_calls.append(str(index[i].date()))
                close_trade(i, price, forced=True)
                blocked = True
                equity = cash

        is_long = states[i]
        if not is_long:
            blocked = False
        resized = False
        if is_long and not in_pos and not blocked:
            lev, _ = target_at(i, equity)
            adj = price * (1.0 + costs.slippage)
            investable = equity * lev - costs.commission
            if investable > 0 and adj > 0 and equity > 0:
                shares = investable / adj
                cash -= shares * adj + costs.commission
                turnover += shares * adj
                in_pos, applied, entry_lev = True, lev, lev
                entry_idx, entry_price, entry_equity = i, adj, equity
                entry_regimes = regimes_at(i)
                last_up_idx = i
                lev_sum, lev_days = 0.0, 0
        elif not is_long and in_pos:
            close_trade(i, price)
        elif is_long and in_pos and daily:
            lev, _ = target_at(i, equity)
            if lev > applied:
                lev = min(lev, applied + cfg.max_step_up) if i - last_up_idx >= cfg.step_days else applied
            if abs(lev - applied) >= cfg.min_change - 1e-12:
                target_shares = equity * lev / price if price > 0 else shares
                delta = target_shares - shares
                fill = price * (1.0 + costs.slippage) if delta > 0 else price * (1.0 - costs.slippage)
                cash -= delta * fill + costs.commission
                turnover += abs(delta * fill)
                shares = target_shares
                if lev > applied:
                    last_up_idx = i
                applied = lev
                resizes += 1
                resized = True
                last_resize = i

        equity = cash + shares * price
        peak = max(peak, equity)
        equity_arr[i] = equity
        lev_arr[i] = (shares * price / equity) if (in_pos and equity > 0) else 0.0
        applied_arr[i] = applied if in_pos else 0.0
        if in_pos:
            lev_sum += lev_arr[i]
            lev_days += 1

    # Diagnóstico del último día: L vigente (o el de una entrada hoy / la próxima entrada)
    if n:
        eq_last = equity_arr[-1]
        if layered:
            target_lev, detail = evaluator.target(n - 1, closed_ret, eq_last, peak, detail=True)
            last_info = {"leverage_policy": POLICY_LAYERED, "apply": cfg.apply, **(detail or {}),
                         "applied_leverage": applied if in_pos else target_lev,
                         "target_leverage": target_lev, "resized_today": last_resize == n - 1,
                         "margin_calls": len(margin_calls)}
        else:
            target_lev, diag = target_at(n - 1, eq_last)
            last_info = {**(diag or {}), "applied_leverage": applied if in_pos else target_lev}
    applied_final = applied if in_pos else (last_info.get("applied_leverage", base_leverage) if n else base_leverage)
    return SimResult(
        dates=index, equity=equity_arr, leverage=lev_arr, applied=applied_arr, trades=trades, margin_calls=margin_calls,
        interest_paid=interest_paid, resizes=resizes, turnover=turnover, in_position=in_pos,
        applied_leverage=float(applied_final), last_info=last_info, resized_last_day=last_resize == n - 1,
    )


def live_layered_leverage(closes, long_state, dates, policy, inputs=None, regime_labels=None,
                          slippage_pct: float = DEFAULT_SLIPPAGE_PCT, costs: Optional[SimCosts] = None):
    """Apalancamiento layered de HOY para un motor en vivo: mismo simulador que el backtest (paridad)."""
    if dates is None:
        raise ValueError("La politica layered necesita las fechas de las barras.")
    costs = costs or SimCosts(slippage=slippage_pct)
    res = simulate(closes, long_state, dates, policy, inputs=inputs, regime_labels=regime_labels, costs=costs)
    return res.applied_leverage, {**res.last_info, "leverage_resized": res.resized_last_day}
