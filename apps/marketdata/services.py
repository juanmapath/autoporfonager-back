import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from decimal import Decimal
import pandas as pd
from django.utils import timezone as dj_timezone

from apps.instruments.models import Instrument
from apps.marketdata.models import Bar, LatestQuote
from apps.marketdata.providers.yahoo import YahooProvider

logger = logging.getLogger(__name__)

FULL_HISTORY_START = datetime(1970, 1, 2, tzinfo=timezone.utc)
INCREMENTAL_SYNC_DAYS = 60


class MarketDataService:
    def __init__(self, provider=None):
        self.provider = provider or YahooProvider()

    def sync_bars(self, symbol: str, days: Optional[int] = 730) -> int:
        """
        Fetches and upserts historical daily bars for the given instrument symbol.
        `days=None` downloads the full history available from the provider.
        """
        instrument, _ = Instrument.objects.get_or_create(symbol=symbol)
        if days is None:
            start_dt = FULL_HISTORY_START
        else:
            start_dt = dj_timezone.now() - timedelta(days=days)
        yahoo_sym = instrument.yahoo_symbol or symbol

        try:
            df = self.provider.get_daily_bars(yahoo_sym, start_dt)
        except Exception as e:
            logger.error(f"Failed to fetch daily bars for {symbol}: {e}")
            return 0

        def dec(val, fallback, places=6):
            return Decimal(str(round(val if pd.notnull(val) else fallback, places)))

        now = dj_timezone.now()
        bars_by_date = {}
        for _, row in df.iterrows():
            d = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
            close = row["close"]
            bars_by_date[d] = Bar(
                instrument=instrument,
                date=d,
                open=dec(row["open"], close),
                high=dec(row["high"], close),
                low=dec(row["low"], close),
                close=dec(close, close),
                adj_close=dec(row["adj_close"], close),
                volume=dec(row["volume"], 0, places=2),
                updated_at=now,
            )

        Bar.objects.bulk_create(
            bars_by_date.values(),
            batch_size=1000,
            update_conflicts=True,
            unique_fields=["instrument", "date"],
            update_fields=["open", "high", "low", "close", "adj_close", "volume", "updated_at"],
        )
        return len(bars_by_date)

    def ensure_history(self, instrument: Instrument, incremental: bool = True) -> int:
        """
        Guarantees the instrument has its full available history: downloads it once and marks the
        instrument, then (if `incremental`) only refreshes the most recent bars on later calls.
        """
        if not instrument.full_history_synced:
            count = self.sync_bars(instrument.symbol, days=None)
            if count > 0:
                instrument.full_history_synced = True
                instrument.save(update_fields=["full_history_synced"])
            return count
        if incremental:
            return self.sync_bars(instrument.symbol, days=INCREMENTAL_SYNC_DAYS)
        return 0

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
