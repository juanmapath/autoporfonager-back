from typing import Dict, Any, List, Optional
from decimal import Decimal
import numpy as np
import pandas as pd
from apps.strategies.signals.catalog import STRATEGY_MAP


class BacktestEngine:
    """
    Standardized Backtesting Engine utilizing the centralized strategy signal functions.
    Ensures 100% parity between research/backtesting results and live trading signals.
    """

    def __init__(self, initial_cash: float = 100000.0, commission_per_trade: float = 0.0):
        self.initial_cash = float(initial_cash)
        self.commission = float(commission_per_trade)

    def run(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        params: Any,
        rebalance_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs backtest against OHLCV DataFrame using the specified strategy from STRATEGY_MAP.
        """
        strat_func = STRATEGY_MAP.get(strategy_name)
        if not strat_func:
            raise ValueError(f"Strategy '{strategy_name}' not found in STRATEGY_MAP.")

        closes = df["Close"].copy()
        signals, indicators = strat_func(df, params)

        cash = self.initial_cash
        shares = 0.0
        portfolio_values = []
        trades = []
        entry_price = 0.0
        entry_idx = None

        n = len(df)
        for i in range(n):
            price = float(closes.iloc[i]) if pd.notnull(closes.iloc[i]) else 0.0
            raw_sig = signals.iloc[i] if i < len(signals) else 0
            sig = int(raw_sig) if pd.notnull(raw_sig) else 0

            # Execute Signal
            if sig == 1 and shares == 0:
                # Buy signal
                investable = cash - self.commission
                if investable > 0 and price > 0:
                    shares = investable / price
                    cash = 0.0
                    entry_price = price
                    entry_idx = i
            elif sig == -1 and shares > 0:
                # Sell signal
                proceeds = (shares * price) - self.commission
                pnl = proceeds - (shares * entry_price)
                pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0.0
                trades.append({
                    "entry_index": entry_idx,
                    "exit_index": i,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                })
                cash = proceeds
                shares = 0.0
                entry_price = 0.0

            curr_val = cash + (shares * price)
            portfolio_values.append(curr_val)

        pv_series = pd.Series(portfolio_values, index=df.index)
        
        # Calculate Metrics
        final_val = float(pv_series.iloc[-1]) if not pv_series.empty else self.initial_cash
        total_roi = ((final_val - self.initial_cash) / self.initial_cash) * 100

        # Max Drawdown
        cum_max = pv_series.cummax()
        drawdown = (pv_series - cum_max) / cum_max
        max_dd = float(drawdown.min() * 100) if not drawdown.empty else 0.0

        # CAGR
        days = (df.index[-1] - df.index[0]).days if hasattr(df.index[0], "day") and len(df) > 1 else len(df)
        years = max(days / 365.25, 0.1)
        cagr = (((final_val / self.initial_cash) ** (1 / years)) - 1) * 100 if final_val > 0 else 0.0

        # Sharpe
        daily_ret = pv_series.pct_change().dropna()
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)) if not daily_ret.empty and daily_ret.std() > 0 else 0.0

        # Trade Stats
        n_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0

        total_gain = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = round(total_gain / total_loss, 2) if total_loss > 0 else (999.0 if total_gain > 0 else 0.0)

        # Equity Curve samples (sampled to ~150 points for JSON serialization in Next.js charts)
        step = max(1, len(pv_series) // 150)
        curve = []
        for i in range(0, len(pv_series), step):
            dt_str = str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i])
            curve.append({
                "date": dt_str,
                "value": round(float(pv_series.iloc[i]), 2),
            })

        return {
            "initial_cash": self.initial_cash,
            "final_value": round(final_val, 2),
            "roi_pct": round(total_roi, 2),
            "cagr_pct": round(cagr, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": profit_factor,
            "equity_curve": curve,
            "trades": trades[-20:],  # last 20 trades
        }
