import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional
from decimal import Decimal
import pandas as pd
from django.conf import settings
from django.utils import timezone as dj_timezone

from apps.instruments.models import Instrument
from apps.marketdata.models import Bar, LatestQuote
from apps.marketdata.providers.yahoo import YahooProvider

logger = logging.getLogger(__name__)

FULL_HISTORY_START = datetime(1970, 1, 2, tzinfo=timezone.utc)
INCREMENTAL_SYNC_DAYS = 60
FETCH_RETRIES = 2
FETCH_BACKOFF_SECONDS = (0.25, 0.5)
OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _with_retries(fn: Callable, retries: int = FETCH_RETRIES, backoff=FETCH_BACKOFF_SECONDS):
    """Ejecuta `fn` con hasta `retries` reintentos y backoff corto; re-lanza el último error."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff[min(attempt, len(backoff) - 1)])


class MarketDataService:
    def __init__(self, provider=None, provider_factory: Optional[Callable] = None):
        self.provider = provider or YahooProvider()
        # Para descargas concurrentes: un provider (y su requests.Session) por hilo
        self.provider_factory = provider_factory or ((lambda: provider) if provider else YahooProvider)

    # --- red: descarga sin tocar la BD (apto para hilos) ---

    @staticmethod
    def bars_start(full_history: bool) -> datetime:
        return FULL_HISTORY_START if full_history else dj_timezone.now() - timedelta(days=INCREMENTAL_SYNC_DAYS)

    # --- BD: guardado desde el hilo principal ---

    def store_bars(self, instrument: Instrument, df: pd.DataFrame) -> int:
        """Upsert de barras diarias del DataFrame del provider. Devuelve el número de barras."""
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

    def store_quote(self, instrument: Instrument, q: Dict) -> LatestQuote:
        quote, _ = LatestQuote.objects.update_or_create(
            instrument=instrument,
            defaults={
                "price": Decimal(str(round(q["price"], 6))),
                "timestamp": q["timestamp"],
                "currency": q.get("currency", "USD"),
            }
        )
        return quote

    # --- API por instrumento (serie) ---

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
        return self.store_bars(instrument, df)

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
            return self.store_quote(instrument, q)
        except Exception as e:
            logger.error(f"Failed to update latest quote for {symbol}: {e}")
            return None

    # --- refresco concurrente del universo (paso ① del pipeline) ---

    def refresh_many(
        self,
        instruments: Iterable[Instrument],
        max_workers: Optional[int] = None,
        budget_seconds: Optional[float] = None,
    ) -> Dict:
        """
        Igual que `ensure_history(inst)` + `update_latest_quote(inst.symbol)` por instrumento, pero
        con las descargas HTTP en un ThreadPoolExecutor (un provider por hilo, reintentos cortos) y
        las escrituras en la BD desde el hilo principal. Al vencer `budget_seconds` los instrumentos
        sin terminar se registran como error y el paso continúa.
        """
        instruments = list(instruments)
        max_workers = max_workers or getattr(settings, "MARKETDATA_MAX_WORKERS", 8)
        budget_seconds = budget_seconds if budget_seconds is not None else getattr(settings, "MARKETDATA_STEP_BUDGET_SECONDS", 150)
        by_id = {inst.id: inst for inst in instruments}
        local = threading.local()

        def fetch(inst_id: int, yahoo_sym: str, full_history: bool):
            provider = getattr(local, "provider", None)
            if provider is None:
                provider = local.provider = self.provider_factory()
            start_dt = self.bars_start(full_history)
            bars = quote = None
            errors = []
            try:
                bars = _with_retries(lambda: provider.get_daily_bars(yahoo_sym, start_dt))
            except Exception as e:
                errors.append(f"bars: {e}")
            try:
                quote = _with_retries(lambda: provider.get_latest_quote(yahoo_sym))
            except Exception as e:
                errors.append(f"quote: {e}")
            return inst_id, bars, quote, errors

        errors: Dict[str, List[str]] = {}
        finished = set()
        deadline = time.monotonic() + budget_seconds
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="marketdata")
        futures = [
            executor.submit(fetch, inst.id, inst.yahoo_symbol or inst.symbol, not inst.full_history_synced)
            for inst in instruments
        ]
        try:
            for fut in as_completed(futures, timeout=max(0.0, deadline - time.monotonic())):
                inst_id, bars, quote, errs = fut.result()
                inst = by_id[inst_id]
                try:
                    if bars is not None:
                        count = self.store_bars(inst, bars)
                        if not inst.full_history_synced and count > 0:
                            inst.full_history_synced = True
                            inst.save(update_fields=["full_history_synced"])
                    if quote is not None:
                        self.store_quote(inst, quote)
                except Exception as e:
                    errs.append(f"store: {e}")
                if errs:
                    logger.error(f"Market data refresh failed for {inst.symbol}: {errs}")
                    errors[inst.symbol] = errs
                finished.add(inst_id)
        except FuturesTimeout:
            logger.error(f"Market data refresh: presupuesto de {budget_seconds}s agotado.")
        finally:
            # Las descargas en curso terminan solas (no tocan la BD); las encoladas se cancelan.
            executor.shutdown(wait=False, cancel_futures=True)

        timed_out = [inst.symbol for inst in instruments if inst.id not in finished]
        for symbol in timed_out:
            errors[symbol] = ["timeout: step budget exceeded"]
        return {
            "instruments": len(instruments),
            "errors": len(errors),
            "timed_out": len(timed_out),
            "error_symbols": sorted(errors)[:50],
        }


def load_daily_bars_frames(instrument_ids: Iterable[int], chunk_size: int = 50) -> Dict[int, pd.DataFrame]:
    """
    Caché de barras para el cálculo de señales: {instrument_id: DataFrame Date→Open/High/Low/Close/Volume
    en float}, idéntico al que antes se construía por bot e instrumento, pero leyendo la BD en lote
    (una consulta por bloque de `chunk_size` instrumentos, para acotar memoria con historiales largos).
    Los instrumentos sin barras no aparecen.
    """
    ids = sorted(set(instrument_ids))
    frames: Dict[int, pd.DataFrame] = {}
    numeric = OHLCV_COLUMNS[1:]
    for start in range(0, len(ids), chunk_size):
        rows = list(
            Bar.objects.filter(instrument_id__in=ids[start:start + chunk_size])
            .order_by("instrument_id", "date")
            .values_list("instrument_id", "date", "open", "high", "low", "close", "volume")
        )
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["instrument_id"] + OHLCV_COLUMNS)
        df[numeric] = df[numeric].astype(float)
        for inst_id, group in df.groupby("instrument_id", sort=False):
            frames[int(inst_id)] = group[OHLCV_COLUMNS].set_index("Date")
    return frames
