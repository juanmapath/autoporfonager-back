"""Datos de referencia de las capas (VIX, VIX3M, HYG, LQD, canasta de turbulencia, ^IRX...) para vivo y backtest."""
import logging
from typing import Dict, Iterable, List, Optional

import pandas as pd

from apps.strategies.leverage.config import required_symbols
from apps.strategies.leverage.layers import LayerInputs
from apps.strategies.leverage.simulator import RATE_SYMBOL
from apps.strategies.signals.sizing import POLICY_LAYERED, LeveragePolicy, resolve_leverage_policy

logger = logging.getLogger(__name__)

INDEX_LIKE = {"^VIX", "^VIX3M", "^IRX", "^NDX", "DX-Y.NYB"}
_BACKTEST_CACHE: Dict[tuple, pd.DataFrame] = {}


def policy_reference_symbols(policy: LeveragePolicy) -> List[str]:
    if policy.type != POLICY_LAYERED:
        return []
    syms = required_symbols(policy.layered)
    if RATE_SYMBOL not in syms:
        syms.append(RATE_SYMBOL)  # interés de margen: mismo coste en vivo y backtest
    return syms


def reference_symbols_for_versions(versions: Iterable) -> List[str]:
    out: List[str] = []
    for v in versions:
        try:
            policy = resolve_leverage_policy(v.params)
        except ValueError:
            continue
        for s in policy_reference_symbols(policy):
            if s not in out:
                out.append(s)
    return out


def ensure_reference_instruments(symbols: Iterable[str]):
    from apps.instruments.models import Instrument

    out = []
    for symbol in symbols:
        inst, _ = Instrument.objects.get_or_create(
            symbol=symbol,
            defaults={"name": symbol, "asset_class": "index" if symbol in INDEX_LIKE else "etf", "tradable": False},
        )
        out.append(inst)
    return out


def reference_instrument_ids() -> set:
    """Instrumentos de referencia de las capas que usa algún bot live activo (entran al universo)."""
    from apps.strategies.models import StrategyVersion

    versions = StrategyVersion.objects.all().filter(status="live", strategy__is_active=True).only("params")
    symbols = reference_symbols_for_versions(versions)
    return {i.id for i in ensure_reference_instruments(symbols)} if symbols else set()


def inputs_from_frames(frames_by_symbol: Dict[str, pd.DataFrame]) -> LayerInputs:
    closes = {}
    for sym, df in frames_by_symbol.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].astype(float).copy()
        s.index = pd.to_datetime(s.index)
        closes[sym] = s[~s.index.duplicated(keep="last")].sort_index()
    return LayerInputs(closes=closes)


def fetch_backtest_frames(symbols: Iterable[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Barras de Yahoo para el backtest (caché por proceso: el estudio corre muchas configuraciones)."""
    from apps.strategies.backtest import fetch_historical_ohlcv

    out = {}
    for sym in symbols:
        key = (sym, start_date, end_date)
        if key not in _BACKTEST_CACHE:
            try:
                _BACKTEST_CACHE[key] = fetch_historical_ohlcv(sym, start_date, end_date)
            except ValueError as e:
                logger.warning(f"Capas: sin datos de {sym}: {e}")
                _BACKTEST_CACHE[key] = pd.DataFrame()
        out[sym] = _BACKTEST_CACHE[key]
    return out


def backtest_inputs(policy: LeveragePolicy, start_date: str, end_date: str,
                    extra: Optional[Iterable[str]] = None) -> LayerInputs:
    symbols = policy_reference_symbols(policy) + [s for s in (extra or []) if s]
    return inputs_from_frames(fetch_backtest_frames(dict.fromkeys(symbols), start_date, end_date))
