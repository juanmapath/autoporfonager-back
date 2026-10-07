import requests
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd


class YahooProvider:
    """
    Robust Yahoo Finance market data provider using HTTP requests (no Selenium).
    Retrieves daily OHLCV bars and latest intraday quotes.
    """
    BASE_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.update(self.HEADERS)

    def get_daily_bars(self, symbol: str, start_dt: datetime) -> pd.DataFrame:
        """Downloads historical daily bars from start_dt to now."""
        p1 = int(start_dt.timestamp())
        p2 = int(datetime.now(timezone.utc).timestamp())
        url = self.BASE_CHART_URL.format(symbol=symbol)
        params = {
            "period1": p1,
            "period2": p2,
            "interval": "1d",
            "events": "div|split",
            "includeAdjustedClose": "true",
        }
        res = self.session.get(url, params=params, timeout=15)
        res.raise_for_status()
        return self._parse_chart_json(res.json(), symbol)

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Gets current regular market price and timestamp."""
        url = self.BASE_CHART_URL.format(symbol=symbol)
        params = {"range": "1d", "interval": "1m"}
        res = self.session.get(url, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()["chart"]["result"][0]
        meta = data["meta"]
        return {
            "price": meta["regularMarketPrice"],
            "timestamp": datetime.fromtimestamp(meta["regularMarketTime"], tz=timezone.utc),
            "currency": meta.get("currency", "USD"),
        }

    def _parse_chart_json(self, raw_json: dict, symbol: str) -> pd.DataFrame:
        result = raw_json["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        quote = result["indicators"]["quote"][0]
        adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adj,
            "volume": quote.get("volume"),
        })
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df
