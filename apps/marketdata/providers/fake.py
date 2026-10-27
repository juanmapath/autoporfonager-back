import time
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd


class FakeProvider:
    """
    Provider sin red para tests y benchmark: misma interfaz que YahooProvider, con barras
    diarias deterministas (paseo aleatorio sembrado por símbolo) y latencia simulada por request.
    """

    def __init__(self, latency_ms: float = 0, seed: int = 0, history_years: int = 10):
        self.latency_ms = latency_ms
        self.seed = seed
        self.history_days = int(history_years * 365)
        self.calls = 0

    def _sleep(self):
        self.calls += 1
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)

    def _rng(self, symbol: str) -> np.random.Generator:
        return np.random.default_rng(zlib.crc32(symbol.encode()) + self.seed)

    def _base_price(self, symbol: str) -> float:
        return 20 + (zlib.crc32(symbol.encode()) % 400)

    def get_daily_bars(self, symbol: str, start_dt: datetime) -> pd.DataFrame:
        self._sleep()
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = max(start_dt, end - timedelta(days=self.history_days))
        dates = pd.bdate_range(start.date(), end.date(), tz="UTC")
        # El paseo se genera desde una fecha fija para que la descarga incremental coincida
        # con la completa en los días que se solapan.
        anchor = pd.bdate_range(end.date() - timedelta(days=self.history_days), end.date(), tz="UTC")
        rets = self._rng(symbol).normal(0.0004, 0.015, len(anchor))
        closes = pd.Series(self._base_price(symbol) * np.exp(np.cumsum(rets)), index=anchor).reindex(dates)
        closes = closes.dropna()
        dates = closes.index
        close = closes.values
        return pd.DataFrame({
            "timestamp": [int(d.timestamp()) for d in dates],
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(len(close), 1_000_000.0),
            "Date": dates,
        })

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        self._sleep()
        return {
            "price": float(self._base_price(symbol)),
            "timestamp": datetime.now(timezone.utc),
            "currency": "USD",
        }
