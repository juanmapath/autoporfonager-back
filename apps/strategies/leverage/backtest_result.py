"""Resultado de backtest (mismo formato que BacktestEngine._simulate) para la política layered."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from apps.strategies.leverage.metrics import cagr, max_drawdown, sharpe
from apps.strategies.leverage.simulator import SimCosts, SimResult, simulate


def run_layered_backtest(
    engine, df_eval: pd.DataFrame, signals_eval: pd.Series, symbol: str, policy, inputs,
    regime_labels, entry_meta_by_date: Optional[Dict[str, List[str]]] = None,
    exit_meta_by_date: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    state = (signals_eval.fillna(0) > 0).values
    res: SimResult = simulate(
        df_eval["Close"].values, state, df_eval.index, policy, inputs=inputs, regime_labels=regime_labels,
        costs=SimCosts(slippage=engine.slippage, commission=engine.commission), initial_cash=engine.initial_cash,
    )
    return format_result(engine.initial_cash, df_eval, symbol, res, policy, entry_meta_by_date, exit_meta_by_date)


def format_result(initial_cash, df_eval, symbol, res: SimResult, policy, entry_meta=None, exit_meta=None):
    dates = df_eval.index
    closes = df_eval["Close"].values.astype(float)
    pv = pd.Series(res.equity, index=dates)
    bh = pd.Series(initial_cash / closes[0] * closes, index=dates)
    final_val, bh_final = float(pv.iloc[-1]), float(bh.iloc[-1])

    trades = []
    markers = []
    for t in res.trades:
        e_meta = (entry_meta or {}).get(t["entry_date"])
        x_meta = (exit_meta or {}).get(t["exit_date"])
        item = {
            "entry_date": t["entry_date"], "exit_date": t["exit_date"],
            "entry_price": round(t["entry_price"], 2), "exit_price": round(t["exit_price"], 2),
            "pnl": round(t["pnl"], 2), "pnl_pct": round(t["return_pct"], 2),
            "leverage": round(t["leverage_entry"], 2), "avg_leverage": round(t["avg_leverage"], 2),
            "entry_strategy": " + ".join(e_meta) if e_meta else "Estrategia",
            "exit_strategy": "Llamada de margen" if t["forced"] else (" + ".join(x_meta) if x_meta else "Regla de Salida"),
            "forced": t["forced"],
        }
        trades.append(item)
        markers.append({"date": item["entry_date"], "type": "BUY", "price": item["entry_price"],
                        "strategy": item["entry_strategy"], "leverage": item["leverage"]})
        markers.append({"date": item["exit_date"], "type": "SELL", "price": item["exit_price"],
                        "strategy": item["exit_strategy"], "pnl": item["pnl"], "pnl_pct": item["pnl_pct"],
                        "is_win": item["pnl"] > 0})
    if res.in_position and len(dates):
        markers.append({"date": str(dates[-1].date()), "type": "OPEN", "price": round(float(closes[-1]), 2),
                        "strategy": "En Posición (Abierto)", "leverage": round(res.applied_leverage, 2)})

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gain, loss = sum(t["pnl"] for t in wins), abs(sum(t["pnl"] for t in losses))
    r = pv.pct_change().dropna()
    downside = r[r < 0]
    dstd = float(np.sqrt(np.mean(downside ** 2))) if not downside.empty else 0.0
    step = max(1, len(pv) // 160)
    curve = [{"date": str(dates[i].date()), "strategy_value": round(float(pv.iloc[i]), 2),
              "benchmark_value": round(float(bh.iloc[i]), 2)} for i in range(0, len(pv), step)]
    if curve and curve[-1]["date"] != str(dates[-1].date()):
        curve.append({"date": str(dates[-1].date()), "strategy_value": round(final_val, 2),
                      "benchmark_value": round(bh_final, 2)})
    roi = (final_val / initial_cash - 1) * 100
    bh_roi = (bh_final / initial_cash - 1) * 100
    return {
        "symbol": symbol, "initial_cash": initial_cash, "final_value": round(final_val, 2),
        "benchmark_final_value": round(bh_final, 2), "roi_pct": round(roi, 2), "benchmark_roi_pct": round(bh_roi, 2),
        "cagr_pct": round(cagr(pv) * 100, 2), "benchmark_cagr_pct": round(cagr(bh) * 100, 2),
        "max_drawdown_pct": round(max_drawdown(pv) * 100, 2), "benchmark_max_drawdown_pct": round(max_drawdown(bh) * 100, 2),
        "sharpe_ratio": round(sharpe(pv), 2),
        "sortino_ratio": round(float(r.mean() / dstd * np.sqrt(252)) if dstd > 0 else 0.0, 2),
        "total_trades": len(trades), "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gain / loss, 2) if loss > 0 else (99.0 if gain > 0 else 0.0),
        "alpha_vs_benchmark": round(roi - bh_roi, 2), "equity_curve": curve, "trades": trades,
        "trade_markers": markers,
        "margin_calls": res.margin_calls, "margin_interest_paid": round(res.interest_paid, 2),
        "resizes": res.resizes,
        "applied_leverage": {
            "base_leverage": policy.layered.base, "max_leverage": policy.layered.max, "use_dynamic_leverage": True,
            "leverage_policy": policy.to_dict(), "current": round(res.applied_leverage, 4),
        },
    }
