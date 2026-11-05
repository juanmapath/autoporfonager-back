"""
Servicios de regímenes: instrumentos requeridos, series de etiquetas (misma función en vivo y en
backtest), alineación con las barras del activo operado y persistencia diaria (RegimeSnapshot).
"""
import json
import logging
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from apps.strategies.regimes import REGIME_PROVIDERS
from apps.strategies.signals.sizing import LeveragePolicy, resolve_leverage_policy

logger = logging.getLogger(__name__)

INDEX_SYMBOLS = {"^VIX": "index", "DX-Y.NYB": "index"}
SYMBOL_NAMES = {
    "HYG": "iShares iBoxx High Yield Corporate Bond ETF", "LQD": "iShares iBoxx Investment Grade Corporate Bond ETF",
    "DX-Y.NYB": "US Dollar Index (DXY)", "^VIX": "CBOE Volatility Index (VIX)",
}


def policy_providers(policy: LeveragePolicy) -> List[str]:
    return [p for p in policy.providers if p in REGIME_PROVIDERS] if policy.uses_regimes else []


def provider_keys_for_versions(versions: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    """{proveedor: params del proveedor} que usan las versiones dadas con política regime_pf."""
    keys: Dict[str, Dict[str, Any]] = {}
    for v in versions:
        policy = resolve_leverage_policy(v.params)
        for key in policy_providers(policy):
            keys.setdefault(key, dict(policy.provider_params.get(key) or {}))
    return keys


def required_symbols(provider_keys: Iterable[str]) -> List[str]:
    symbols: List[str] = []
    for key in provider_keys:
        for s in REGIME_PROVIDERS[key].required_symbols():
            if s not in symbols:
                symbols.append(s)
    return symbols


def ensure_regime_instruments(provider_keys: Iterable[str]):
    """Instrumentos no operables de los proveedores (historial completo vía prewarm/marketdata)."""
    from apps.instruments.models import Instrument

    out = []
    for symbol in required_symbols(provider_keys):
        inst, _ = Instrument.objects.get_or_create(
            symbol=symbol,
            defaults={"name": SYMBOL_NAMES.get(symbol, symbol), "asset_class": INDEX_SYMBOLS.get(symbol, "etf"),
                      "tradable": False},
        )
        out.append(inst)
    return out


def active_regime_provider_keys() -> Dict[str, Dict[str, Any]]:
    from apps.strategies.models import StrategyVersion

    versions = StrategyVersion.objects.all().filter(status="live", strategy__is_active=True).only("params")
    return provider_keys_for_versions(versions)


def regime_instrument_ids() -> set:
    """Instrumentos macro de los proveedores usados por algún bot live activo (entran al universo)."""
    keys = active_regime_provider_keys()
    if not keys:
        return set()
    return {i.id for i in ensure_regime_instruments(keys)}


def compute_label_series(provider_key: str, bars: Dict[str, pd.DataFrame], params: Optional[dict] = None) -> pd.Series:
    """Serie de etiquetas del proveedor (index DatetimeIndex). Misma función en vivo y backtest."""
    return REGIME_PROVIDERS[provider_key].compute(bars, params or {})


def align_labels(series: Optional[pd.Series], index) -> List[Optional[str]]:
    """Etiqueta vigente en cada fecha de `index` (último valor conocido hasta esa fecha, causal)."""
    if series is None or len(series) == 0:
        return [None] * len(index)
    target = pd.to_datetime(pd.Index(index))
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    union = s.index.union(target)
    aligned = s.reindex(union).ffill().reindex(target)
    return [None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v) for v in aligned.tolist()]


def policy_and_labels(params: Optional[dict], index):
    """(política, {proveedor: etiquetas alineadas a `index`} o None) para un motor o el backtest."""
    policy = resolve_leverage_policy(params)
    if not policy.uses_regimes:
        return policy, None
    series = (params or {}).get("_regime_series") or {}
    return policy, {p: align_labels(series.get(p), index) for p in policy.providers}


def build_series_from_bars(provider_keys: Dict[str, Dict[str, Any]], bars_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    return {key: compute_label_series(key, bars_by_symbol, prm) for key, prm in provider_keys.items()}


def series_for_policy(policy: LeveragePolicy, bars_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    keys = {p: dict(policy.provider_params.get(p) or {}) for p in policy_providers(policy)}
    return build_series_from_bars(keys, bars_by_symbol)


def fetch_series_backtest(policy: LeveragePolicy, start_date: str, end_date: str) -> Dict[str, pd.Series]:
    """Serie de régimen para el backtest: descarga las barras de los símbolos requeridos (Yahoo)."""
    from apps.strategies.backtest import fetch_historical_ohlcv

    bars: Dict[str, pd.DataFrame] = {}
    for symbol in required_symbols(policy_providers(policy)):
        bars[symbol] = fetch_historical_ohlcv(symbol, start_date, end_date)
    return series_for_policy(policy, bars)


def save_snapshots(provider_keys: Dict[str, Dict[str, Any]], bars_by_symbol: Dict[str, pd.DataFrame], as_of=None) -> Dict[str, str]:
    """
    Calcula y guarda el régimen del día de cada proveedor en uso (antes de las señales). Devuelve
    {proveedor: etiqueta}. Un proveedor sin datos suficientes no genera snapshot.
    """
    from apps.strategies.models import RegimeSnapshot

    out: Dict[str, str] = {}
    for key, prm in provider_keys.items():
        provider = REGIME_PROVIDERS[key]
        series = provider.compute(bars_by_symbol, prm).dropna()
        if series.empty:
            logger.warning(f"Regimen '{key}' sin datos suficientes: no se guarda snapshot.")
            continue
        day = pd.Timestamp(series.index[-1]).date()
        features = provider.features(bars_by_symbol, prm) if hasattr(provider, "features") else {}
        RegimeSnapshot.objects.update_or_create(
            provider=key, date=day,
            defaults={"label": str(series.iloc[-1]), "features": json.loads(json.dumps(features, default=str))},
        )
        out[key] = str(series.iloc[-1])
    return out
