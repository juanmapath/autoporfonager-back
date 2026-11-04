"""
Informe de regimenes de una version (comando `regime_report`): por regimen, trades, PF, retorno medio,
apalancamiento resultante y duracion media; y comparacion de backtests con las cuatro politicas.
"""
from typing import Any, Dict, List, Optional

import pandas as pd

from apps.strategies.regimes.service import align_labels, fetch_series_backtest
from apps.strategies.signals.sizing import (
    ClosedTrade, LeveragePolicy, POLICY_REGIME_PF, policy_entry_leverage, profit_factor, resolve_leverage_policy,
)

COMPARISON = ("fixed_base", "fixed_max", "global_pf", "regime_pf")
LABELS = {
    "fixed_base": "Fijo (base)", "fixed_max": "Fijo (max)", "global_pf": "PF global", "regime_pf": "PF por regimen",
}


def regime_durations(series: pd.Series) -> Dict[str, Dict[str, float]]:
    """Duracion media (dias de mercado y dias calendario) de cada racha continua de un regimen."""
    s = series.dropna()
    out: Dict[str, Dict[str, float]] = {}
    if s.empty:
        return out
    idx = pd.to_datetime(s.index)
    labels = s.astype(str).values
    start = 0
    runs: Dict[str, List[tuple]] = {}
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            end_cal = idx[i] if i < len(labels) else idx[-1] + pd.Timedelta(days=1)
            runs.setdefault(labels[start], []).append((i - start, (end_cal - idx[start]).days))
            start = i
    for label, rs in runs.items():
        out[label] = {
            "runs": len(rs),
            "avg_market_days": sum(r[0] for r in rs) / len(rs),
            "avg_calendar_days": sum(r[1] for r in rs) / len(rs),
            "pct_of_time": 100.0 * sum(r[0] for r in rs) / len(labels),
        }
    return out


def _metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    trades = res.get("trades", [])
    levs = [t.get("leverage", 1.0) for t in trades]
    return {
        "cagr_pct": res["cagr_pct"], "max_drawdown_pct": res["max_drawdown_pct"], "sharpe": res["sharpe_ratio"],
        "roi_pct": res["roi_pct"], "trades": res["total_trades"], "profit_factor": res["profit_factor"],
        "avg_leverage": round(sum(levs) / len(levs), 3) if levs else None,
    }


def build_regime_report(
    version, provider: str = "macro", start_date: str = "1999-03-10", end_date: Optional[str] = None,
    initial_cash: float = 100000.0,
) -> Dict[str, Any]:
    from apps.strategies.backtest import BacktestEngine

    params = version.params or {}
    base = float(params.get("leverage") or 1.0)
    max_lev = max(base, float(params.get("max_leverage") or base))
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    policy = resolve_leverage_policy(params)
    if policy.type != POLICY_REGIME_PF or provider not in policy.providers:
        policy = LeveragePolicy(type=POLICY_REGIME_PF, providers=(provider,))
    series_by_provider = fetch_series_backtest(policy, start_date, end_date)
    series = series_by_provider[provider]

    engine = BacktestEngine(initial_cash=initial_cash)
    runs: Dict[str, Dict[str, Any]] = {}
    plans = {
        "fixed_base": (base, base, {"type": "fixed"}),
        "fixed_max": (max_lev, max_lev, {"type": "fixed"}),
        "global_pf": (base, max_lev, {"type": "global_pf"}),
        "regime_pf": (base, max_lev, policy.to_dict()),
    }
    for name, (lev, mx, pol) in plans.items():
        runs[name] = engine.run_for_strategy(
            version.strategy, start_date=start_date, end_date=end_date, leverage=lev, max_leverage=mx,
            leverage_policy=pol, regime_series=series_by_provider, version=version,
        )

    # Trades del backtest con apalancamiento fijo (base) etiquetados con el regimen de su entrada
    trades = [t for t in runs["fixed_base"]["trades"] if not t.get("open")]
    entry_labels = align_labels(series, [pd.Timestamp(t["entry_date"]) for t in trades])
    closed = [
        ClosedTrade(pd.Timestamp(t["entry_date"]), pd.Timestamp(t["exit_date"]), t["pnl"], {provider: lab})
        for t, lab in zip(trades, entry_labels)
    ]
    as_of = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    durations = regime_durations(series)

    rows = []
    for label in sorted({c.regimes[provider] for c in closed if c.regimes[provider]} | set(durations)):
        in_r = [(c, t) for c, t in zip(closed, trades) if c.regimes[provider] == label]
        lev, info = policy_entry_leverage(policy, closed, base, max_lev, as_of, {provider: label})
        pinfo = info["providers"][provider]
        d = durations.get(label, {})
        rows.append({
            "regime": label,
            "trades": len(in_r),
            "profit_factor": round(profit_factor([c.pnl for c, _ in in_r]), 2) if in_r else None,
            "avg_return_pct": round(sum(t["pnl_pct"] for _, t in in_r) / len(in_r), 2) if in_r else None,
            "pf_shrunk": round(pinfo["pf_shrunk"], 2) if pinfo["pf_shrunk"] is not None else None,
            "leverage": round(lev, 2),
            "source": pinfo["source"],
            "avg_duration_market_days": round(d.get("avg_market_days", 0.0), 1),
            "avg_duration_calendar_days": round(d.get("avg_calendar_days", 0.0), 1),
            "pct_of_time": round(d.get("pct_of_time", 0.0), 1),
        })

    return {
        "bot": version.strategy.name, "version_id": version.id, "provider": provider,
        "period": [start_date, end_date], "base_leverage": base, "max_leverage": max_lev,
        "policy": policy.to_dict(), "regimes": rows,
        "comparison": {name: _metrics(res) for name, res in runs.items()},
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        f"=== {report['bot']} (version {report['version_id']}) | proveedor {report['provider']} | "
        f"{report['period'][0]} -> {report['period'][1]} | lev base {report['base_leverage']} / max {report['max_leverage']}",
        "",
        f"{'Regimen':<10}{'Trades':>7}{'PF':>7}{'Ret.medio%':>12}{'PF*':>7}{'Apalanc.':>10}{'Dur.mkt(d)':>12}"
        f"{'Dur.cal(d)':>12}{'%tiempo':>9}  Fuente",
    ]
    for r in report["regimes"]:
        f = lambda v, nd=2: "-" if v is None else f"{v:.{nd}f}"
        lines.append(
            f"{r['regime']:<10}{r['trades']:>7}{f(r['profit_factor']):>7}{f(r['avg_return_pct']):>12}"
            f"{f(r['pf_shrunk']):>7}{r['leverage']:>10.2f}{r['avg_duration_market_days']:>12.1f}"
            f"{r['avg_duration_calendar_days']:>12.1f}{r['pct_of_time']:>9.1f}  {r['source']}"
        )
    lines += ["", f"{'Politica':<18}{'CAGR%':>8}{'MaxDD%':>9}{'Sharpe':>8}{'Trades':>8}{'Lev.medio':>11}"]
    for name in COMPARISON:
        m = report["comparison"][name]
        lines.append(
            f"{LABELS[name]:<18}{m['cagr_pct']:>8.2f}{m['max_drawdown_pct']:>9.2f}{m['sharpe']:>8.2f}"
            f"{m['trades']:>8}{(m['avg_leverage'] if m['avg_leverage'] is not None else 0):>11.2f}"
        )
    return "\n".join(lines)
