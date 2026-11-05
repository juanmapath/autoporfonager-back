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
from apps.strategies.regimes.service import align_labels
from apps.strategies.signals.sizing import (
    ClosedTrade, LeveragePolicy, aligned_label, leverage_from_pf, policy_entry_leverage, resolve_leverage_policy,
)

# Modo hold con regime_pf: PF diario de ganancia/dolor por regimen (umbrales y minimos propios, en dias)
HOLD_MIN_DAYS = 60
HOLD_PF_MAX = 1.35
HOLD_PF_MID = 1.10
HOLD_PF_CAP = 5.0


class RankedAllocationEngine(BaseEngine):
    """
    Ranks multiple assets based on defined metrics and distributes capital exposure.

    Supported parameters in `params`:
      - metrics: List[{"key": str, "weight": float}]
      - rank_weights: List[float] e.g. [0.30, 0.25, 0.20, 0.15, 0.10]
      - frozen_ranking: List[str] (optional) symbols in rank order to reuse instead of re-ranking
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

        # Leverage & Regime parameters
        base_leverage = float(params.get("leverage", 1.0))
        max_leverage = float(params.get("max_leverage", base_leverage))
        # Politica de apalancamiento: fixed, o regime_pf (use_regimes=true o leverage_policy.type="regime_pf")
        policy = resolve_leverage_policy(params)
        regime_series = (params.get("_regime_series") or {}) if policy.uses_regimes else {}

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
        # Missing values (None / NaN / +-inf) rank LAST (pct 0.0), below any asset with data.
        percentiles: Dict[str, Dict[str, float]] = {}
        missing_by_symbol: Dict[str, List[str]] = {sym: [] for sym in symbols}
        for m_cfg in metric_configs:
            m_key = m_cfg.get("key")
            m_def = METRIC_REGISTRY.get(m_key)
            if not m_def or m_key not in raw_values:
                continue

            vals_dict = raw_values[m_key]
            percentiles[m_key] = self._metric_percentiles(vals_dict, symbols, m_def.direction)
            for sym in symbols:
                if not self._is_finite(vals_dict.get(sym)):
                    missing_by_symbol[sym].append(m_key)

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

        # 6. Sort assets by score (descending) with deterministic symbol tie-break.
        # Between rebalance dates the caller passes `frozen_ranking` (order of the last ranking) so
        # daily price moves in the metrics do not reshuffle positions; scores are still reported.
        frozen_ranking = params.get("frozen_ranking") or []
        ranking_frozen = len(frozen_ranking) == n_assets and set(frozen_ranking) == set(symbols)
        if ranking_frozen:
            ranked_symbols = list(frozen_ranking)
        else:
            ranked_symbols = sorted(symbols, key=lambda s: (-composite_scores.get(s, 0.0), s))

        # 7. Map rank to capital weight
        assigned_capital_weights: Dict[str, Decimal] = {}
        for rank_idx, sym in enumerate(ranked_symbols):
            weight = rank_weights[rank_idx] if rank_idx < len(rank_weights) else Decimal("0.0")
            assigned_capital_weights[sym] = weight

        # 8. Execution Layer: Hold vs Strategy + Per-Asset Dynamic Regime Leverage
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
            sigs = None

            if exec_mode == "strategy":
                if underlying_strat_func and df is not None and not df.empty:
                    try:
                        sigs, ind = underlying_strat_func(df, underlying_strat_params)
                        sub_signal = self._resolve_active_signal_state(sigs, ind)
                        if isinstance(ind, pd.DataFrame) and not ind.empty:
                            sub_indicators = {
                                k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.number)) else str(v))
                                for k, v in ind.iloc[-1].to_dict().items()
                            }
                    except Exception:
                        sub_signal = 0
                else:
                    sub_signal = 0

            lev_info: Dict[str, Any] = {"leverage_policy": policy.type, "applied_leverage": base_leverage}
            applied_lev = base_leverage
            if policy.uses_regimes:
                applied_lev, lev_info = self._compute_policy_leverage(
                    df=df, sigs=sigs if exec_mode == "strategy" else None, policy=policy,
                    base_leverage=base_leverage, max_leverage=max_leverage, regime_series=regime_series,
                )

            lev_dec = Decimal("1") if abs(applied_lev - 1.0) < 1e-9 else Decimal(str(round(applied_lev, 4)))

            if exec_mode == "strategy":
                # If strategy is out (sub_signal <= 0), exposure is 0 (stays in cash)
                exposure = (cap_weight * lev_dec) if sub_signal > 0 else Decimal("0.0")
            else:
                # 'hold' mode: exposure scaled by per-asset regime leverage
                exposure = cap_weight * lev_dec

            targets[sym] = exposure

            # Build diagnostics per symbol
            sym_raw_metrics = {m_key: raw_values.get(m_key, {}).get(sym) for m_key in raw_values}
            sym_percentiles = {m_key: percentiles.get(m_key, {}).get(sym) for m_key in percentiles}

            diagnostics[sym] = {
                "rank": rank_idx + 1,
                "ranking_frozen": ranking_frozen,
                "composite_score": round(composite_scores.get(sym, 0.0), 4),
                "assigned_weight": str(cap_weight),
                "applied_leverage": round(applied_lev, 4),
                "leverage": lev_info,
                "target_exposure": str(exposure),
                "execution_mode": exec_mode,
                "sub_strategy_signal": sub_signal,
                "raw_metrics": sym_raw_metrics,
                "percentiles": sym_percentiles,
                "missing_metrics": missing_by_symbol.get(sym, []),
                "sub_indicators": sub_indicators,
            }

        return targets, diagnostics

    @staticmethod
    def _is_finite(v: Any) -> bool:
        if v is None:
            return False
        try:
            return bool(np.isfinite(float(v)))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _metric_percentiles(
        cls,
        values: Dict[str, Optional[float]],
        symbols: List[str],
        direction: str,
    ) -> Dict[str, float]:
        """
        Percentile (1.0 best .. 0.0 worst) per symbol for one metric.

        Assets with data are ordered by `direction` and get pct = 1 - idx / (n_total - 1).
        Missing values (None/NaN/inf) get 0.0 so they always rank below any asset with data.
        Edge cases: no data at all or a single asset -> everyone 0.5 (metric is uninformative).
        """
        n_total = len(symbols)
        valid = [(s, float(values[s])) for s in symbols if cls._is_finite(values.get(s))]

        if n_total <= 1 or not valid:
            return {s: 0.5 for s in symbols}

        # asc: smaller is better; desc: higher is better. Stable sort keeps input order on ties.
        ordered = sorted(valid, key=lambda x: x[1], reverse=(direction == "desc"))
        result = {s: 0.0 for s in symbols}
        for idx, (sym, _) in enumerate(ordered):
            result[sym] = 1.0 - idx / float(n_total - 1)
        return result

    @staticmethod
    def _resolve_active_signal_state(sigs: pd.Series, ind: Any) -> int:
        """
        Determines whether the underlying strategy is currently in an active LONG position (1) or FLAT (0).
        Supports both multi-state pulse strategies (1=entry, 0=hold, -1=exit with SignalType)
        and continuous signal series.
        """
        if sigs is None or sigs.empty:
            return 0

        if isinstance(ind, pd.DataFrame) and "SignalType" in ind.columns and not ind.empty:
            last_type = str(ind["SignalType"].iloc[-1]).lower()
            return 1 if last_type not in ("none", "nan", "") else 0

        # Walk signals chronologically if exit pulses (-1) are used
        if (sigs == -1).any():
            in_pos = False
            for s_val in sigs.values:
                if s_val == 1:
                    in_pos = True
                elif s_val == -1:
                    in_pos = False
            return 1 if in_pos else 0

        return 1 if int(sigs.iloc[-1]) > 0 else 0

    @staticmethod
    def _compute_policy_leverage(
        df: Optional[pd.DataFrame],
        sigs: Optional[pd.Series],
        policy: LeveragePolicy,
        base_leverage: float,
        max_leverage: float,
        regime_series: Dict[str, pd.Series],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apalancamiento por la politica regime_pf (misma `policy_entry_leverage` que el resto de motores).
        - Modo 'strategy' (hay `sigs`): trades de la sub-estrategia en este activo; si hay un trade
          abierto el apalancamiento queda fijado en su ENTRADA (trades cerrados y regimen de ese dia).
        - Modo 'hold' (sin trades): PF diario de ganancia/dolor del activo en los dias del regimen
          vigente (retorno del dia t etiquetado con el regimen del cierre t-1);
          base si no hay `HOLD_MIN_DAYS` dias de historia en ese regimen.
        """
        eff_max = max(base_leverage, max_leverage)
        if df is None or df.empty or "Close" not in df.columns:
            return base_leverage, {"leverage_policy": policy.type, "source": "no_data", "applied_leverage": base_leverage}

        index = pd.to_datetime(pd.Index(df.index))
        labels = {p: align_labels(regime_series.get(p), index) for p in policy.providers}
        closes = df["Close"].astype(float).values

        if sigs is not None and not sigs.empty and len(sigs) == len(df):
            closed: List[ClosedTrade] = []
            in_pos, entry_idx = False, 0
            for i, (s_val, c_val) in enumerate(zip(sigs.values, closes)):
                if s_val == 1 and not in_pos:
                    in_pos, entry_idx = True, i
                elif s_val == -1 and in_pos:
                    in_pos = False
                    if closes[entry_idx] > 0:
                        closed.append(ClosedTrade(
                            index[entry_idx], index[i], (c_val - closes[entry_idx]) / closes[entry_idx],
                            {p: aligned_label(labels[p], entry_idx) for p in policy.providers},
                        ))
            if in_pos:
                at, as_of = entry_idx, index[entry_idx]
            else:
                at, as_of = len(index) - 1, index[-1] + pd.Timedelta(days=1)
            now = {p: aligned_label(labels[p], at) for p in policy.providers}
            lev, info = policy_entry_leverage(policy, closed, base_leverage, eff_max, as_of, now)
            first = next(iter(info.get("providers", {}).values()), {})
            return lev, {**info, "mode": "strategy_trades", "source": first.get("source")}

        # Modo hold: PF diario condicionado al regimen
        rets = pd.Series(closes).pct_change().values
        provider = policy.providers[0]
        lag = [None] + [aligned_label(labels[provider], i) for i in range(len(index) - 1)]  # regimen del cierre t-1
        today = aligned_label(labels[provider], len(index) - 1)
        if policy.lookback_years:
            cutoff = index[-1] - pd.Timedelta(days=int(policy.lookback_years * 365.25))
        else:
            cutoff = index[0]
        valid = [(r_, l_) for r_, l_, d_ in zip(rets, lag, index) if d_ >= cutoff and l_ is not None and np.isfinite(r_)]
        in_regime = [r_ for r_, l_ in valid if l_ == today]

        def daily_pf(values):
            pos = sum(v for v in values if v > 0)
            neg = abs(sum(v for v in values if v < 0))
            return pos / neg if neg > 0 else (HOLD_PF_CAP if pos > 0 else 1.0)

        info = {
            "leverage_policy": policy.type, "mode": "hold_daily_pf", "regime": today,
            "n_trades_regime": len(in_regime), "pf_regime": None, "source": "base_insufficient_trades",
        }
        if today is None:
            info.update(source="base_no_regime", applied_leverage=base_leverage)
            return base_leverage, info
        if len(in_regime) < HOLD_MIN_DAYS:
            info["applied_leverage"] = base_leverage
            return base_leverage, info
        pf_r = min(daily_pf(in_regime), HOLD_PF_CAP)
        lev = leverage_from_pf(pf_r, base_leverage, eff_max, HOLD_PF_MAX, HOLD_PF_MID)
        info.update(pf_regime=pf_r, source="regime_pf", applied_leverage=lev)
        return lev, info
