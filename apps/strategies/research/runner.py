import os
import logging
from typing import Dict, Any, List
import pandas as pd
from core.security_guard import is_local_mode_active
from apps.strategies.signals.catalog import STRATEGY_MAP
from apps.strategies.backtest import BacktestEngine

logger = logging.getLogger(__name__)


class LocalResearchRunner:
    """
    Exclusive local research execution runner.
    Verifies that LOCAL_MODE is active before running heavy parametric scans,
    macro matrix analysis, or regime optimization.
    """

    def __init__(self):
        if not is_local_mode_active():
            logger.warning("LocalResearchRunner is executing while LOCAL_MODE is False. Ensure you are not affecting production.")

    def run_parameter_grid(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        param_grid: List[Dict[str, Any]],
        initial_cash: float = 100000.0,
    ) -> pd.DataFrame:
        """
        Runs parametric grid search over a strategy and ranks parameter combinations by Sharpe and ROI.
        """
        engine = BacktestEngine(initial_cash=initial_cash)
        results = []

        for p in param_grid:
            try:
                res = engine.run(df, strategy_name, p)
                results.append({
                    "params": str(p),
                    "roi_pct": res["roi_pct"],
                    "cagr_pct": res["cagr_pct"],
                    "max_drawdown_pct": res["max_drawdown_pct"],
                    "sharpe_ratio": res["sharpe_ratio"],
                    "win_rate_pct": res["win_rate_pct"],
                    "profit_factor": res["profit_factor"],
                    "total_trades": res["total_trades"],
                })
            except Exception as e:
                logger.error(f"Error evaluating params {p}: {e}")

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.sort_values(by="sharpe_ratio", ascending=False).reset_index(drop=True)
        return res_df
