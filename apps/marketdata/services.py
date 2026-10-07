import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pandas as pd
from django.utils import timezone as dj_timezone

from apps.instruments.models import Instrument
from apps.marketdata.models import Bar, LatestQuote
from apps.marketdata.providers.yahoo import YahooProvider

logger = logging.getLogger(__name__)


class MarketDataService:
    def __init__(self, provider=None):
        self.provider = provider or YahooProvider()

    def sync_bars(self, symbol: str, days: int = 180) -> int:
        """Fetches and updates historical daily bars for the given instrument symbol."""
        instrument, _ = Instrument.objects.get_or_create(symbol=symbol)
        start_dt = dj_timezone.now() - timedelta(days=days)
        yahoo_sym = instrument.yahoo_symbol or symbol
        
        try:
            df = self.provider.get_daily_bars(yahoo_sym, start_dt)
        except Exception as e:
            logger.error(f"Failed to fetch daily bars for {symbol}: {e}")
            return 0

        created_count = 0
        for _, row in df.iterrows():
            d = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
            Bar.objects.update_or_create(
                instrument=instrument,
                date=d,
                defaults={
                    "open": Decimal(str(round(row["open"], 6))),
                    "high": Decimal(str(round(row["high"], 6))),
                    "low": Decimal(str(round(row["low"], 6))),
                    "close": Decimal(str(round(row["close"], 6))),
                    "adj_close": Decimal(str(round(row["adj_close"], 6))),
                    "volume": Decimal(str(round(row["volume"], 2))) if pd.notnull(row["volume"]) else Decimal("0.00"),
                }
            )
            created_count += 1
        return created_count

    def update_latest_quote(self, symbol: str) -> Optional[LatestQuote]:
        """Fetches and caches the latest intraday market quote."""
        instrument, _ = Instrument.objects.get_or_create(symbol=symbol)
        yahoo_sym = instrument.yahoo_symbol or symbol
        try:
            q = self.provider.get_latest_quote(yahoo_sym)
            quote, _ = LatestQuote.objects.update_or_create(
                instrument=instrument,
                defaults={
                    "price": Decimal(str(round(q["price"], 6))),
                    "timestamp": q["timestamp"],
                    "currency": q.get("currency", "USD"),
                }
            )
            return quote
        except Exception as e:
            logger.error(f"Failed to update latest quote for {symbol}: {e}")
            return None
