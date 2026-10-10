"""
Passive Hold & Cash Reserve Engine.

Computes signals for structural holding:
- If holding Cash (is_cash=True or symbol='CASH'/'USD'), target exposure is 0.0 (stays liquid).
- If holding an asset (e.g. SPY, BTC), target exposure is 1.0 (fully invested).
- Carries 'participates_in_rebalance' flag in diagnostics to inform PortfolioAllocator.
"""
from decimal import Decimal
from typing import Dict, Any, Tuple
import pandas as pd
from apps.strategies.engines.base import BaseEngine


class HoldEngine(BaseEngine):
    def compute_signal(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        instruments_weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
        is_cash = bool(params.get("is_cash", False))
        participates = bool(params.get("participates_in_rebalance", True))

        targets: Dict[str, Decimal] = {}
        diagnostics: Dict[str, Any] = {}

        for sym, weight in instruments_weights.items():
            if is_cash or sym.upper() in ("CASH", "USD"):
                # Cash reserve: zero equity exposure (funds remain unallocated/cash)
                exposure = Decimal("0.0")
            else:
                # Passive buy & hold: target 100% of the assigned weight
                exposure = weight

            targets[sym] = exposure
            diagnostics[sym] = {
                "is_cash": is_cash,
                "participates_in_rebalance": participates,
                "target_exposure": str(exposure),
                "mode": "hold_reserve",
            }

        return targets, diagnostics
