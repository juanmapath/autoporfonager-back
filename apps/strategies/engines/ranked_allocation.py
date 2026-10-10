"""
Ranked Allocation Engine.

Computes multi-asset ranking based on quantitative / fundamental metrics,
scores each asset in [0.0, 1.0] by percentiles, sorts them deterministically,
assigns capital weights based on user-defined rank allocations,
and evaluates execution either as direct 'hold' or through an underlying
single_signal strategy.
"""
from datetime import date
from decimal import Decimal
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from django.utils import timezone

from apps.strategies.engines.base import BaseEngine
from apps.strategies.signals.catalog import STRATEGY_MAP
from apps.instruments.models import Instrument
from apps.fundamentals.models import FundamentalStatement
from apps.fundamentals.metrics import METRIC_REGISTRY, MetricContext


class RankedAllocationEngine(BaseEngine):
    """
    Ranks multiple assets based on defined metrics and distributes capital exposure.

    Supported parameters in `params`:
      - metrics: List[{"key": str, "weight": float}]
      - rank_weights: List[float] e.g. [0.30, 0.25, 0.20, 0.15, 0.10]
      - execution: {
            "mode": "hold" | "strategy",
            "strategy_name": str (optional, required if mode == "strategy"),
            "params": dict (optional)
        }
    """

    def compute_signal(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        instruments_weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
        metric_configs = params.get("metrics", [])
        rank_weights_input = params.get("rank_weights", [])
        execution = params.get("execution", {})
        exec_mode = execution.get("mode", "hold")
        underlying_strat_name = execution.get("strategy_name") or execution.get("strategy_function")
        underlying_strat_params = execution.get("params", execution.get("strategy_params", {}))

        symbols = list(ohlcv_data.keys())
        n_assets = len(symbols)

        if n_assets == 0:
            return {}, {}

        # 1. Prepare rank weights
        # If the user didn't specify rank_weights or length mismatch, distribute equally
        if len(rank_weights_input) == n_assets:
            rank_weights = [Decimal(str(w)) for w in rank_weights_input]
        else:
            eq_weight = Decimal("1.0") / Decimal(str(n_assets))
            rank_weights = [eq_weight] * n_assets

        # 2. Gather Metric Contexts for each asset
        as_of_today = timezone.now().date()
        asset_contexts: Dict[str, MetricContext] = {}

        for sym, df in ohlcv_data.items():
            inst = Instrument.objects.filter(symbol=sym).first()
            if not inst:
                continue

            # Fetch Latest Price
            price = None
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])

            # Fetch available fundamentals point-in-time
            stmts = list(
                FundamentalStatement.objects.filter(
                    instrument=inst,
                    available_at__lte=as_of_today,
                ).order_by("-period_end")[:12]
            )

            asset_contexts[sym] = MetricContext(
                instrument=inst,
                as_of=as_of_today,
                price=price,
                bars_df=df,
                statements=stmts,
            )

        # 3. Compute raw metric values
        # raw_values[metric_key][symbol] = float or None
        raw_values: Dict[str, Dict[str, Optional[float]]] = {}
        for m_cfg in metric_configs:
            m_key = m_cfg.get("key")
            m_def = METRIC_REGISTRY.get(m_key)
            if not m_def:
                continue

            raw_values[m_key] = {}
            for sym in symbols:
                ctx = asset_contexts.get(sym)
                if ctx:
                    try:
                        raw_values[m_key][sym] = m_def.fn(ctx)
                    except Exception:
                        raw_values[m_key][sym] = None
                else:
                    raw_values[m_key][sym] = None

        # 4. Compute percentiles per metric
        # Missing values receive neutral percentile = 0.5
        percentiles: Dict[str, Dict[str, float]] = {}
        for m_cfg in metric_configs:
            m_key = m_cfg.get("key")
            m_def = METRIC_REGISTRY.get(m_key)
            if not m_def or m_key not in raw_values:
                continue

            vals_dict = raw_values[m_key]
            valid_items = [(sym, v) for sym, v in vals_dict.items() if v is not None and not np.isnan(v)]

            percentiles[m_key] = {}

            if len(valid_items) <= 1:
                # If only 0 or 1 item has data, all get neutral 0.5
                for sym in symbols:
                    percentiles[m_key][sym] = 0.5
            else:
                # Sort according to direction
                # asc: smaller is better -> rank 1 is lowest value -> gets percentile 1.0
                # desc: higher is better -> rank 1 is highest value -> gets percentile 1.0
                reverse_sort = (m_def.direction == "desc")
                sorted_valid = sorted(valid_items, key=lambda x: x[1], reverse=reverse_sort)

                n_valid = len(sorted_valid)
                for idx, (sym, _) in enumerate(sorted_valid):
                    # Percentile from 1.0 (best) to 0.0 (worst)
                    pct = 1.0 - (idx / float(n_valid - 1)) if n_valid > 1 else 0.5
                    percentiles[m_key][sym] = float(pct)

                # Assign neutral 0.5 to those with missing data
                for sym in symbols:
                    if sym not in percentiles[m_key]:
                        percentiles[m_key][sym] = 0.5

        # 5. Composite Score per Asset
        composite_scores: Dict[str, float] = {}
        total_metric_weight = sum(float(m.get("weight", 1.0)) for m in metric_configs) or 1.0

        for sym in symbols:
            score = 0.0
            if metric_configs:
                for m_cfg in metric_configs:
                    m_key = m_cfg.get("key")
                    w = float(m_cfg.get("weight", 1.0))
                    pct = percentiles.get(m_key, {}).get(sym, 0.5)
                    score += w * pct
                composite_scores[sym] = score / total_metric_weight
            else:
                # No metrics defined -> equal score
                composite_scores[sym] = 0.5

        # 6. Sort assets by score (descending) with deterministic symbol tie-break
        ranked_symbols = sorted(symbols, key=lambda s: (-composite_scores.get(s, 0.0), s))

        # 7. Map rank to capital weight
        assigned_capital_weights: Dict[str, Decimal] = {}
        for rank_idx, sym in enumerate(ranked_symbols):
            weight = rank_weights[rank_idx] if rank_idx < len(rank_weights) else Decimal("0.0")
            assigned_capital_weights[sym] = weight

        # 8. Execution Layer: Hold vs Strategy
        targets: Dict[str, Decimal] = {}
        diagnostics: Dict[str, Any] = {}

        underlying_strat_func = None
        if exec_mode == "strategy" and underlying_strat_name:
            underlying_strat_func = STRATEGY_MAP.get(underlying_strat_name)

        for rank_idx, sym in enumerate(ranked_symbols):
            cap_weight = assigned_capital_weights[sym]
            df = ohlcv_data.get(sym)

            sub_signal = 1
            sub_indicators = {}

            if exec_mode == "strategy":
                if underlying_strat_func and df is not None and not df.empty:
                    try:
                        sigs, ind = underlying_strat_func(df, underlying_strat_params)
                        sub_signal = int(sigs.iloc[-1]) if not sigs.empty else 0
                        if isinstance(ind, pd.DataFrame) and not ind.empty:
                            sub_indicators = {
                                k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.number)) else str(v))
                                for k, v in ind.iloc[-1].to_dict().items()
                            }
                    except Exception:
                        sub_signal = 0
                else:
                    sub_signal = 0

                # If strategy is out (sub_signal <= 0), exposure is 0 (stays in cash)
                exposure = cap_weight if sub_signal > 0 else Decimal("0.0")
            else:
                # 'hold' mode: direct exposure equal to ranked weight
                exposure = cap_weight

            targets[sym] = exposure

            # Build diagnostics per symbol
            sym_raw_metrics = {m_key: raw_values.get(m_key, {}).get(sym) for m_key in raw_values}
            sym_percentiles = {m_key: percentiles.get(m_key, {}).get(sym) for m_key in percentiles}

            diagnostics[sym] = {
                "rank": rank_idx + 1,
                "composite_score": round(composite_scores.get(sym, 0.0), 4),
                "assigned_weight": str(cap_weight),
                "target_exposure": str(exposure),
                "execution_mode": exec_mode,
                "sub_strategy_signal": sub_signal,
                "raw_metrics": sym_raw_metrics,
                "percentiles": sym_percentiles,
                "sub_indicators": sub_indicators,
            }

        return targets, diagnostics
