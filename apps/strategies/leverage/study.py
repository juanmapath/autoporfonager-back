"""
Estudio de apalancamiento (04e §6): mismas señales para todas las configuraciones, mismo simulador
que en vivo, costes de margen, ventanas de crisis, criterio de aceptación (§1) y selección con
control de sobreajuste (§6.4).

Criterio (§1). Una configuración es ACEPTABLE si:
  1. su caída máxima no supera la del benchmark (comprado y mantenido a 1x) en el mismo periodo;
  2. en cada ventana de crisis cae menos (o igual) que el benchmark en esa ventana;
  3. no tiene llamadas de margen.
Entre las aceptables gana la de mayor CAGR fuera de muestra.
"""
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from apps.strategies.leverage import metrics as M
from apps.strategies.leverage.data import fetch_backtest_frames, inputs_from_frames
from apps.strategies.leverage.layers import LAYERS, LayerInputs
from apps.strategies.leverage.simulator import RATE_SYMBOL, SimCosts, simulate
from apps.strategies.signals.sizing import LeveragePolicy, POLICY_FIXED, resolve_leverage_policy

logger = logging.getLogger(__name__)

DEFAULT_CRISES = [
    ("Burbuja .com", "2000-03-01", "2002-10-31"),
    ("Crisis financiera", "2007-10-01", "2009-03-31"),
    ("Q4 2018", "2018-09-01", "2018-12-31"),
    ("COVID", "2020-02-01", "2020-03-31"),
    ("Bajista 2022", "2021-11-01", "2022-12-31"),
]
ALL_REFERENCE_SYMBOLS = ["^VIX", "^VIX3M", "HYG", "LQD", "SPY", "TLT", "GLD", "UUP", RATE_SYMBOL]


@dataclass
class StudyConfig:
    name: str
    group: str
    policy: Dict[str, Any]
    base: Optional[float] = None
    max: Optional[float] = None
    n_layers: int = 0
    note: str = ""

    def build(self, version_base: float, version_max: float) -> Tuple[LeveragePolicy, float, float]:
        base = version_base if self.base is None else self.base
        lmax = version_max if self.max is None else self.max
        if self.policy.get("type") == "layered":
            raw = {"base": base, "max": lmax, **self.policy}
            pol = resolve_leverage_policy({"leverage": base, "max_leverage": lmax, "leverage_policy": raw})
            return pol, pol.layered.base, pol.layered.max
        if self.policy.get("type") == POLICY_FIXED:
            return LeveragePolicy(type=POLICY_FIXED), base, lmax
        pol = resolve_leverage_policy({"leverage_policy": self.policy})
        return pol, base, lmax


# ------------------------------ datos ------------------------------

def splice_proxy(main: pd.DataFrame, proxy: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """Antes del primer día de `main` usa `proxy` escalado (precios al nivel del primer cierre y volumen
    al nivel medio de los primeros 60 días comunes). Devuelve (serie empalmada, fecha de empalme)."""
    if proxy is None or proxy.empty or main.empty:
        return main, None
    first = main.index[0]
    before = proxy[proxy.index < first]
    common = proxy.index.intersection(main.index)
    if before.empty or common.empty:
        return main, None
    k = float(main.loc[common[0], "Close"]) / float(proxy.loc[common[0], "Close"])
    head = before.copy()
    for col in ("Open", "High", "Low", "Close"):
        head[col] = head[col] * k
    win = common[:60]
    pv = float(proxy.loc[win, "Volume"].mean() or 0)
    head["Volume"] = head["Volume"] * (float(main.loc[win, "Volume"].mean()) / pv) if pv > 0 else main["Volume"].iloc[0]
    return pd.concat([head, main]), first


def crisis_windows(bench: pd.Series, custom: Optional[List[Tuple[str, str, str]]] = None,
                   auto_threshold: float = 0.20) -> List[Dict[str, Any]]:
    """Ventanas por defecto dentro del periodo y caídas del benchmark > 20% detectadas automáticamente."""
    start, end = bench.index[0], bench.index[-1]
    out = []
    for name, a, b in (custom or DEFAULT_CRISES):
        a, b = max(pd.Timestamp(a), start), min(pd.Timestamp(b), end)
        if (b - a).days > 5:
            out.append({"name": name, "start": a, "end": b, "auto": False})
    # automáticas: episodios máximo -> mínimo con caída > umbral
    dd = M.drawdown_series(bench)
    peak_date, in_ep, trough_date, trough = bench.index[0], False, None, 0.0
    for d, v in dd.items():
        if v == 0:
            if in_ep and trough <= -auto_threshold:
                out.append({"name": f"Caída {trough:.0%} {peak_date.date()}", "start": peak_date, "end": trough_date,
                            "auto": True})
            peak_date, in_ep, trough = d, False, 0.0
        else:
            in_ep = True
            if v < trough:
                trough, trough_date = v, d
    if in_ep and trough <= -auto_threshold:
        out.append({"name": f"Caída {trough:.0%} {peak_date.date()}", "start": peak_date, "end": trough_date, "auto": True})
    # sin duplicar ventanas automáticas que ya cubre una por defecto
    final = []
    for w in out:
        if w["auto"] and any(not o["auto"] and o["start"] <= w["end"] and w["start"] <= o["end"] for o in out):
            continue
        final.append(w)
    return final


# ------------------------------ evaluación ------------------------------

def _rebase(s: pd.Series, start, end) -> pd.Series:
    seg = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return seg / float(seg.iloc[0]) if len(seg) else seg


def evaluate_window(equity: pd.Series, bench: pd.Series, start, end, crises, margin_calls: List[str]) -> Dict[str, Any]:
    eq, bm = _rebase(equity, start, end), _rebase(bench, start, end)
    s, b = M.summary(eq), M.summary(bm)
    reasons = []
    if s["max_dd"] < b["max_dd"] - 1e-9:
        reasons.append(f"caída máx. {s['max_dd']:.1%} peor que el benchmark {b['max_dd']:.1%}")
    crisis_rows = []
    for w in crises:
        if w["end"] < pd.Timestamp(start) or w["start"] > pd.Timestamp(end):
            continue
        a, z = max(w["start"], pd.Timestamp(start)), min(w["end"], pd.Timestamp(end))
        dd_c, dd_b = M.window_drawdown(eq, a, z), M.window_drawdown(bm, a, z)
        if dd_c is None or dd_b is None:
            continue
        ok = dd_c >= dd_b - 1e-9
        crisis_rows.append({"name": w["name"], "dd": dd_c, "bench_dd": dd_b, "ok": ok})
        if not ok:
            reasons.append(f"{w['name']}: {dd_c:.1%} vs benchmark {dd_b:.1%}")
    calls = [c for c in margin_calls if pd.Timestamp(start) <= pd.Timestamp(c) <= pd.Timestamp(end)]
    if calls:
        reasons.append(f"{len(calls)} llamada(s) de margen")
    return {**s, "bench": b, "crises": crisis_rows, "margin_calls": len(calls), "acceptable": not reasons,
            "reasons": reasons}


@dataclass
class ConfigResult:
    config: StudyConfig
    full: Dict[str, Any]
    half_a: Dict[str, Any]
    half_b: Dict[str, Any]
    equity: pd.Series
    applied: pd.Series
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def acceptable_both(self) -> bool:
        return self.half_a["acceptable"] and self.half_b["acceptable"] and self.full["acceptable"]

    @property
    def oos_score(self) -> float:
        """CAGR fuera de muestra: promedio del CAGR de cada mitad, que es la muestra de validación de la otra."""
        return (self.half_a["cagr"] + self.half_b["cagr"]) / 2


class LeverageStudy:
    def __init__(self, version, start: str = "2007-01-01", end: Optional[str] = None,
                 split: str = "2017-01-01", proxy: Optional[Dict[str, str]] = None,
                 costs: Optional[SimCosts] = None, initial_cash: float = 100000.0, history_start: str = "1985-01-01",
                 traded_symbol: Optional[str] = None):
        self.version = version
        self.start, self.split = pd.Timestamp(start), pd.Timestamp(split)
        self.end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        self.proxy = proxy or {}
        self.costs = costs or SimCosts()
        self.initial_cash = initial_cash
        self.history_start = history_start
        params = (version.params or {}) if version is not None else {}
        self.base = float(params.get("leverage") or 1.0)
        self.max = max(self.base, float(params.get("max_leverage") or self.base))
        self.traded_symbol = traded_symbol
        self.n_runs = 0
        self.proxy_until = None
        if version is not None:
            self._prepare()

    @classmethod
    def from_data(cls, closes: pd.Series, state, base: float, max_leverage: float, start: str, split: str,
                  end: Optional[str] = None, inputs: Optional[LayerInputs] = None, regime_labels=None,
                  costs: Optional[SimCosts] = None, symbol: str = "X", initial_cash: float = 100000.0):
        """Estudio sobre datos dados (tests y análisis sin descargar nada)."""
        st = cls(None, start=start, end=end or str(closes.index[-1].date()), split=split, costs=costs,
                 initial_cash=initial_cash)
        st.base, st.max, st.symbol = base, max(base, max_leverage), symbol
        st.index = pd.to_datetime(closes.index)
        st.state = np.asarray(state, dtype=bool)
        st.bench_full = pd.Series(closes.values.astype(float), index=st.index)
        st.inputs = LayerInputs(closes={**((inputs.closes if inputs else {})), "self": st.bench_full})
        st.regime_labels = regime_labels or {}
        bench = st.bench_full[(st.index >= st.start) & (st.index <= st.end)]
        st.bench = bench / float(bench.iloc[0])
        st.crises = crisis_windows(st.bench)
        st._second_markets_enabled = False
        return st

    # ---- señales una vez ----
    def _prepare(self):
        from apps.strategies.backtest import BacktestEngine, fetch_historical_ohlcv
        from apps.strategies.regimes.service import align_labels, fetch_series_backtest

        v = self.version
        params = v.params or {}
        insts = list(v.instruments.all().select_related("instrument"))
        traded = next((i for i in insts if i.role == "traded"), insts[0] if insts else None)
        signal = next((i for i in insts if i.role == "signal_source"), None)
        sym = (self.traded_symbol or traded.instrument.symbol).upper()
        sig_sym = signal.instrument.symbol if signal else sym
        end = self.end.strftime("%Y-%m-%d")
        df = fetch_historical_ohlcv(sym, self.history_start, end)
        self.proxy_until = None
        if sym in self.proxy:
            df, self.proxy_until = splice_proxy(df, fetch_historical_ohlcv(self.proxy[sym], self.history_start, end))
        self.symbol = sym
        out = BacktestEngine(initial_cash=self.initial_cash).run_for_config(
            bot_type=params.get("bot_type") or v.strategy.kind, traded_symbol=sym, signal_symbol=sig_sym,
            strategies=params.get("strategies", []), start_date=self.history_start, end_date=end,
            strategy_params=params, signals_only=True, df_traded_override=df,
        )
        self.df = out["df_traded"]
        self.state = (out["signals"] > 0).values
        self.index = pd.to_datetime(self.df.index)
        closes = self.df["Close"].astype(float)
        self.bench_full = pd.Series(closes.values, index=self.index)
        frames = fetch_backtest_frames(ALL_REFERENCE_SYMBOLS, self.history_start, end)
        self.inputs = inputs_from_frames(frames)
        self.inputs.closes["self"] = pd.Series(closes.values, index=self.index)
        self.regime_series = fetch_series_backtest(
            resolve_leverage_policy({"leverage_policy": {"type": "regime_pf"}}), self.history_start, end)
        self.regime_labels = {p: align_labels(s, self.index) for p, s in self.regime_series.items()}
        bench = self.bench_full[(self.index >= self.start) & (self.index <= self.end)]
        self.bench = bench / float(bench.iloc[0])
        self.crises = crisis_windows(self.bench)

    def run_config(self, cfg: StudyConfig) -> ConfigResult:
        policy, base, lmax = cfg.build(self.base, self.max)
        res = simulate(self.bench_full.values, self.state, self.index, policy, base, lmax, inputs=self.inputs,
                       regime_labels=self.regime_labels, costs=self.costs, initial_cash=self.initial_cash,
                       peak_reset_date=self.start)
        self.n_runs += 1
        eq = pd.Series(res.equity, index=self.index)
        eq = eq[(eq.index >= self.start) & (eq.index <= self.end)]
        applied = pd.Series(res.applied, index=self.index)
        applied = applied[(applied.index >= self.start) & (applied.index <= self.end)]
        full = evaluate_window(eq, self.bench, self.start, self.end, self.crises, res.margin_calls)
        a = evaluate_window(eq, self.bench, self.start, self.split - pd.Timedelta(days=1), self.crises, res.margin_calls)
        b = evaluate_window(eq, self.bench, self.split, self.end, self.crises, res.margin_calls)
        invested = applied[applied > 0]
        years = max((self.end - self.start).days / 365.25, 0.1)
        eq_val = eq  # en $ (initial_cash)
        stats = {
            "avg_leverage": float(invested.mean()) if len(invested) else 0.0,
            "pct_days_over_1x": float((applied > 1.0 + 1e-9).mean() * 100),
            "pct_days_invested": float((applied > 0).mean() * 100),
            "turnover_per_year": res.turnover / max(float(eq_val.mean()), 1e-9) / years,
            "interest_paid_pct": res.interest_paid / self.initial_cash * 100,
            "resizes": res.resizes, "margin_calls": len(res.margin_calls),
        }
        return ConfigResult(cfg, full, a, b, eq, applied, stats)

    # ---- grilla y selección ----
    def run_grid(self, configs: List[StudyConfig]) -> Dict[str, Any]:
        results: List[ConfigResult] = []
        staged: List[StudyConfig] = []
        for cfg in configs:
            if cfg.policy.get("extends_best"):
                staged.append(cfg)
            else:
                results.append(self.run_config(cfg))
        # etapa 2: combinaciones que parten de la "mejor" capa de un grupo, elegida SOLO en la mitad A
        for cfg in staged:
            resolved = self._resolve_staged(cfg, results)
            if resolved:
                results.append(self.run_config(resolved))
        selection = self.select(results)
        return {"results": results, "selection": selection, "n_configs": len(results), "n_runs": self.n_runs}

    def _best_in_sample(self, results: List[ConfigResult], predicate) -> Optional[ConfigResult]:
        pool = [r for r in results if predicate(r)]
        acc = [r for r in pool if r.half_a["acceptable"]] or pool
        return max(acc, key=lambda r: r.half_a["cagr"]) if acc else None

    def _resolve_staged(self, cfg: StudyConfig, results: List[ConfigResult]) -> Optional[StudyConfig]:
        spec = cfg.policy["extends_best"]
        if spec == "multi":
            best = self._best_in_sample(results, lambda r: r.config.n_layers >= 2 and r.config.policy.get("type") == "layered")
        else:
            best = self._best_in_sample(results, lambda r: r.config.group == spec)
        if best is None:
            return None
        pol = copy.deepcopy(best.config.policy)
        for k in ("scalers", "caps", "label_policies"):
            pol[k] = list(pol.get(k) or []) + list(cfg.policy.get("add_" + k) or [])
        for k, v in cfg.policy.items():
            if k not in ("extends_best", "add_scalers", "add_caps", "add_label_policies"):
                pol[k] = v
        n = sum(len(pol.get(k) or []) for k in ("scalers", "caps", "label_policies"))
        name = cfg.name.replace("{best}", best.config.name)
        return StudyConfig(name, cfg.group, pol, cfg.base, cfg.max, n, note=f"parte de {best.config.name} (mejor en 2007–{self.split.year - 1})")

    def select(self, results: List[ConfigResult]) -> Dict[str, Any]:
        acc_a = [r for r in results if r.half_a["acceptable"]]
        acc_b = [r for r in results if r.half_b["acceptable"]]
        pick_a = max(acc_a, key=lambda r: r.half_a["cagr"]) if acc_a else None
        pick_b = max(acc_b, key=lambda r: r.half_b["cagr"]) if acc_b else None
        candidates = sorted([r for r in results if r.acceptable_both], key=lambda r: -r.oos_score)
        log: List[str] = []
        if not candidates:
            return {"winner": None, "pick_a": pick_a, "pick_b": pick_b, "candidates": [], "log":
                    ["Ninguna configuración es aceptable en ambas mitades y en el periodo completo."]}
        best_score = candidates[0].oos_score
        # simplicidad: con menos capas y a menos de 1 punto anual de la mejor, preferir la simple
        near = [r for r in candidates if best_score - r.oos_score <= 0.01]
        simple = min(near, key=lambda r: (r.config.n_layers, -r.oos_score))
        if simple is not candidates[0]:
            log.append(f"Simplicidad: {simple.config.name} ({simple.config.n_layers} capas) está a "
                       f"{(best_score - simple.oos_score) * 100:.2f} pts de {candidates[0].config.name} y se prefiere.")
        ordered = [simple] + [r for r in candidates if r is not simple]
        winner, robustness, second = None, None, None
        for cand in ordered[:6]:
            robustness = self.robustness(cand)
            second = self.second_markets(cand)
            if robustness["fragile"]:
                log.append(f"{cand.config.name}: frágil a ±20% ({robustness['reason']}); se pasa a la siguiente.")
                continue
            if second and any(not m["ok"] for m in second):
                log.append(f"{cand.config.name}: viola el límite de caída en " +
                           ", ".join(m["symbol"] for m in second if not m["ok"]) + "; se pasa a la siguiente.")
                continue
            winner = cand
            break
        if winner is None:
            winner = ordered[0]
            log.append("Ningún candidato pasó todos los controles; se reporta el mejor con advertencias.")
        return {"winner": winner, "pick_a": pick_a, "pick_b": pick_b, "candidates": candidates,
                "robustness": robustness, "second_markets": second, "log": log}

    def robustness(self, r: ConfigResult) -> Dict[str, Any]:
        """Cada parámetro numérico de las capas ×0.8 y ×1.2: frágil si el CAGR fuera de muestra cae > 2 pts
        o deja de ser aceptable."""
        pol = r.config.policy
        variants = []
        if pol.get("type") == "layered":
            for kind in ("scalers", "caps", "label_policies"):
                for idx, layer in enumerate(pol.get(kind) or []):
                    layer = layer if isinstance(layer, dict) else {"key": layer}
                    params = {**LAYERS[layer["key"]].defaults, **(layer.get("params") or {})}
                    for pk, pv in params.items():
                        # topes de salida (cap) y la constante de suavizado EWMA no son parámetros de ajuste
                        if isinstance(pv, bool) or not isinstance(pv, (int, float)) or pk in ("cap", "cap_below", "lambda"):
                            continue
                        for f in (0.8, 1.2):
                            newp = copy.deepcopy(pol)
                            item = newp[kind][idx]
                            item = item if isinstance(item, dict) else {"key": item}
                            val = pv * f
                            if pk == "p":
                                val = min(val, 0.99)
                            if isinstance(pv, int) and not isinstance(pv, bool):
                                val = max(1, int(round(val)))
                            item["params"] = {**(item.get("params") or {}), pk: val}
                            newp[kind][idx] = item
                            variants.append((f"{layer['key']}.{pk}×{f}", newp))
        rows, fragile, reason = [], False, ""
        for label, newp in variants:
            vr = self.run_config(StudyConfig(r.config.name + " " + label, "robustez", newp, r.config.base,
                                             r.config.max, r.config.n_layers))
            drop = r.oos_score - vr.oos_score
            ok = vr.acceptable_both and drop <= 0.02
            rows.append({"variant": label, "oos_cagr": vr.oos_score, "drop": drop, "acceptable": vr.acceptable_both,
                         "ok": ok})
            if not ok and not fragile:
                fragile, reason = True, f"{label}: {'no aceptable' if not vr.acceptable_both else f'-{drop * 100:.1f} pts'}"
        return {"rows": rows, "fragile": fragile, "reason": reason}

    def second_markets(self, r: ConfigResult, symbols=("SPY", "IWM")) -> List[Dict[str, Any]]:
        out = []
        if self.version is None or not getattr(self, "_second_markets_enabled", True):
            return out
        for sym in symbols:
            try:
                other = LeverageStudy(self.version, start=self.start.strftime("%Y-%m-%d"), end=self.end.strftime("%Y-%m-%d"),
                                      split=self.split.strftime("%Y-%m-%d"), costs=self.costs,
                                      initial_cash=self.initial_cash, history_start=self.history_start,
                                      traded_symbol=sym)
                res = other.run_config(r.config)
                self.n_runs += 1
                out.append({"symbol": sym, "cagr": res.full["cagr"], "max_dd": res.full["max_dd"],
                            "bench_dd": res.full["bench"]["max_dd"], "ok": res.full["acceptable"],
                            "reasons": res.full["reasons"]})
            except Exception as e:  # sin datos del segundo mercado
                logger.warning(f"Segundo mercado {sym}: {e}")
                out.append({"symbol": sym, "ok": True, "error": str(e)})
        return out
