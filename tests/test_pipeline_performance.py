import io
import threading
import time
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from apps.accounting.models import LogicalPosition
from apps.instruments.models import Instrument
from apps.marketdata.models import Bar, LatestQuote
from apps.marketdata.providers.fake import FakeProvider
from apps.marketdata.services import MarketDataService, load_daily_bars_frames
from apps.strategies.engines import get_engine
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument, Signal
from apps.trading import tasks
from apps.trading.pipeline import get_or_create_today_pipeline
from apps.trading.universe import pipeline_universe
from core.market_time import get_ny_now
from tests.test_pipeline_instrumentation import _fake_provider
from tests.test_pipeline_safety import _paper_portfolio


# ---------- helpers ----------

def _instrument(symbol, synced=True, **kw):
    inst = Instrument.objects.create(symbol=symbol, name=symbol, **kw)
    if synced:
        MarketDataService(provider=FakeProvider()).ensure_history(inst, incremental=False)
        inst.refresh_from_db()
    return inst


def _bot(slug, engine, kind, params, instruments, status="live", active=True):
    strategy = Strategy.objects.create(name=slug, slug=slug, kind=kind, engine=engine, is_active=active)
    version = StrategyVersion.objects.create(strategy=strategy, version=1, status=status, params=params)
    for inst, role in instruments:
        StrategyInstrument.objects.create(version=version, instrument=inst, role=role, weight=Decimal("1.0"))
    return version


def _fixture_bots():
    """Bots con los engines y parámetros reales del entorno (dev DB) más un cross_asset y un hold."""
    goog, qqq, ibit, tlt, aapl, msft = (_instrument(s) for s in ("GOOG", "QQQ", "IBIT", "TLT", "AAPL", "MSFT"))
    return [
        _bot("mc10-goog-zcrosdema", "single_signal", "one_strategy", {
            "strategies": [{"strategy_name": "Combo_ZCrossDema", "params": [14.0, 90.0, 112.0, 28.0, 3.0, 30.0]}],
            "leverage": 1.0, "max_leverage": 1.5, "use_regimes": True,
        }, [(goog, "traded")]),
        _bot("kvobull-weakrsi", "multi_signal", "multi_strategy", {
            "strategies": [
                {"strategy_name": "MeanRev_WeakRSI", "params": [10.0, 30.0, 10.0]},
                {"strategy_name": "TrendFollowing_KVOBullSPT", "params": [7.0, 28.0, 28.0, 2.0]},
            ],
            "leverage": 1.0, "max_leverage": 1.7, "use_regimes": True,
        }, [(qqq, "traded")]),
        _bot("btc-accumulator", "smart_accumulator", "smart_accumulator", {
            "strategies": [{"strategy_name": "Momentum_Zpullback", "params": [20.0, -1.5]}],
            "leverage": 1.0, "max_leverage": 1.0,
        }, [(ibit, "traded")]),
        _bot("tlt-qqq-cross", "cross_asset", "cross_asset", {
            "strategies": [{"strategy_name": "TrendFollowing_GoldCross", "params": [50, 200]}],
        }, [(qqq, "traded"), (tlt, "signal_source")]),
        _bot("hold-qqq", "hold", "hold", {}, [(qqq, "traded")]),
        _bot("ranked-mega", "ranked_allocation", "ranked_allocation", {
            "metrics": [{"key": "dist_52w_low", "weight": 1.0}], "execution": {"mode": "hold"},
        }, [(goog, "traded"), (aapl, "traded"), (msft, "traded")]),
    ]


def _legacy_ohlcv(version):
    """Lectura de barras del método anterior: una consulta por bot e instrumento."""
    ohlcv = {}
    for si in version.instruments.all().select_related("instrument"):
        rows = list(
            Bar.objects.filter(instrument=si.instrument).order_by("date")
            .values_list("date", "open", "high", "low", "close", "volume")
        )
        if rows:
            df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")
            ohlcv[si.instrument.symbol] = df.astype(float)
    return ohlcv


def _legacy_targets(version):
    """Señales calculadas exactamente como antes de H3 (sin caché), para comparar."""
    instruments = version.instruments.all().select_related("instrument")
    weights = {si.instrument.symbol: si.weight for si in instruments}
    params = dict(version.params or {})
    if not params.get("source_symbol"):
        source = next((si for si in instruments if si.role == "signal_source"), None)
        if source:
            params["source_symbol"] = source.instrument.symbol
    targets, _ = get_engine(version.strategy.engine).compute_signal(_legacy_ohlcv(version), params, weights)
    return targets


# ---------- 2. Universo acotado ----------

@pytest.mark.django_db
def test_marketdata_universe_excludes_instruments_without_use():
    used = Instrument.objects.create(symbol="USED")
    source = Instrument.objects.create(symbol="^VIX", tradable=False)
    held = Instrument.objects.create(symbol="HELD")
    retired_only = Instrument.objects.create(symbol="RETIRED")
    inactive_only = Instrument.objects.create(symbol="INACTIVE")
    Instrument.objects.create(symbol="UNUSED")
    closed = Instrument.objects.create(symbol="CLOSED")

    _bot("live", "hold", "hold", {}, [(used, "traded"), (source, "signal_source")])
    _bot("old", "hold", "hold", {}, [(retired_only, "traded")], status="retired")
    _bot("off", "hold", "hold", {}, [(inactive_only, "traded")], active=False)
    portfolio = _paper_portfolio()
    strategy = Strategy.objects.get(slug="old")
    LogicalPosition.objects.create(portfolio=portfolio, strategy=strategy, instrument=held, qty=Decimal("3"))
    LogicalPosition.objects.create(portfolio=portfolio, strategy=strategy, instrument=closed, qty=Decimal("0"))

    assert set(pipeline_universe().values_list("symbol", flat=True)) == {"USED", "^VIX", "HELD"}

    with _fake_provider():
        assert tasks.marketdata_refresh() == "Refreshed 3 instruments"
    assert set(LatestQuote.objects.values_list("instrument__symbol", flat=True)) == {"USED", "^VIX", "HELD"}


# ---------- 3. HTTP concurrente en ① ----------

@pytest.mark.django_db
def test_concurrent_refresh_writes_from_main_thread_and_matches_serial_download():
    a, b = _instrument("AAA", synced=False), _instrument("BBB", synced=False)
    store_threads = []
    real_store = MarketDataService.store_bars

    def spy_store(self, instrument, df):
        store_threads.append(threading.current_thread() is threading.main_thread())
        return real_store(self, instrument, df)

    with patch.object(MarketDataService, "store_bars", spy_store):
        result = MarketDataService(provider_factory=FakeProvider).refresh_many([a, b], max_workers=4)

    assert result["errors"] == 0 and result["instruments"] == 2
    assert store_threads == [True, True]
    for inst in (a, b):
        inst.refresh_from_db()
        assert inst.full_history_synced
        assert LatestQuote.objects.filter(instrument=inst).exists()
        expected = FakeProvider().get_daily_bars(inst.symbol, MarketDataService.bars_start(True))
        assert Bar.objects.filter(instrument=inst).count() == len(expected)


@pytest.mark.django_db
def test_refresh_budget_marks_remaining_instruments_as_errors_and_returns():
    instruments = [_instrument(f"S{i}", synced=False) for i in range(6)]
    started = time.monotonic()
    result = MarketDataService(provider_factory=lambda: FakeProvider(latency_ms=300)).refresh_many(
        instruments, max_workers=1, budget_seconds=0.2,
    )
    assert time.monotonic() - started < 1.5
    assert result["timed_out"] >= 5
    assert result["errors"] == result["timed_out"]


@pytest.mark.django_db
def test_refresh_retries_transient_http_errors():
    inst = _instrument("FLAKY", synced=False)

    class Flaky(FakeProvider):
        failures = {"bars": 1, "quote": 2}

        def get_daily_bars(self, symbol, start_dt):
            if Flaky.failures["bars"]:
                Flaky.failures["bars"] -= 1
                raise ConnectionError("reset")
            return super().get_daily_bars(symbol, start_dt)

        def get_latest_quote(self, symbol):
            if Flaky.failures["quote"]:
                Flaky.failures["quote"] -= 1
                raise ConnectionError("reset")
            return super().get_latest_quote(symbol)

    with patch("apps.marketdata.services.time.sleep"):
        result = MarketDataService(provider_factory=Flaky).refresh_many([inst])
    assert result["errors"] == 0
    assert LatestQuote.objects.filter(instrument=inst).exists()


# ---------- 4 y 5. Caché de barras y señales sin N+1: equivalencia ----------

@pytest.mark.django_db
def test_bars_cache_frames_are_identical_to_per_bot_reads():
    versions = _fixture_bots()
    ids = {si.instrument_id for v in versions for si in v.instruments.all()}
    cache = load_daily_bars_frames(ids, chunk_size=2)
    for v in versions:
        for symbol, legacy_df in _legacy_ohlcv(v).items():
            inst_id = Instrument.objects.get(symbol=symbol).id
            pd.testing.assert_frame_equal(cache[inst_id], legacy_df)


@pytest.mark.django_db
def test_signals_with_bars_cache_are_identical_to_previous_method():
    versions = _fixture_bots()
    expected = {}
    for v in versions:
        for symbol, exposure in _legacy_targets(v).items():
            expected[(v.strategy.slug, symbol)] = exposure
    assert expected, "el fixture debe producir señales"

    tasks.compute_regimes_and_signals()

    got = {
        (s.version.strategy.slug, s.instrument.symbol): s.target_exposure
        for s in Signal.objects.select_related("version__strategy", "instrument")
    }
    assert set(got) == set(expected)
    for key, exposure in expected.items():
        # Mismo valor que guardaba el método anterior (DecimalField de 4 decimales)
        assert got[key] == Decimal(exposure).quantize(Decimal("0.0001")), key
    assert all(s.changed for s in Signal.objects.all())


@pytest.mark.django_db
def test_changed_flag_uses_preloaded_last_signal_without_n_plus_one():
    versions = _fixture_bots()
    tasks.compute_regimes_and_signals()
    first_ids = set(Signal.objects.values_list("id", flat=True))

    # Semántica anterior: changed = (última señal guardada, cuantizada) != exposición cruda del engine
    previous = {(s.version_id, s.instrument.symbol): s.target_exposure
                for s in Signal.objects.select_related("instrument")}
    expected_changed = {
        (v.id, symbol): previous[(v.id, symbol)] != exposure
        for v in versions for symbol, exposure in _legacy_targets(v).items()
    }
    assert True in expected_changed.values() and False in expected_changed.values()

    with CaptureQueriesContext(connection) as ctx:
        tasks.compute_regimes_and_signals(force=True)

    second = Signal.objects.exclude(id__in=first_ids).select_related("instrument")
    assert {(s.version_id, s.instrument.symbol): s.changed for s in second} == expected_changed
    signal_reads = [q["sql"] for q in ctx.captured_queries
                    if q["sql"].lstrip().upper().startswith("SELECT") and '"strategies_signal"' in q["sql"]]
    assert len(signal_reads) <= 2, signal_reads  # no una consulta por bot e instrumento
    bar_reads = [q for q in ctx.captured_queries if '"marketdata_bar"' in q["sql"]]
    assert len(bar_reads) == 1  # 9 instrumentos en un bloque


@pytest.mark.django_db
def test_latest_signal_exposures_returns_most_recent_per_version_and_instrument():
    v = _bot("h", "hold", "hold", {}, [])
    a, b = Instrument.objects.create(symbol="A"), Instrument.objects.create(symbol="B")
    now = timezone.now()
    for inst, exposure, minutes in ((a, "0.5", 10), (a, "1.0", 5), (b, "0.25", 1)):
        Signal.objects.create(version=v, instrument=inst, as_of=now - pd.Timedelta(minutes=minutes),
                              target_exposure=Decimal(exposure))
    assert tasks.latest_signal_exposures([v.id]) == {(v.id, a.id): Decimal("1.0"), (v.id, b.id): Decimal("0.25")}


# ---------- 1. Sin descarga de historial completo dentro de ② ----------

@pytest.mark.django_db
def test_signals_step_never_downloads_full_history():
    new = _instrument("NEW", synced=False)
    old = _instrument("OLD")
    _bot("new-bot", "hold", "hold", {}, [(new, "traded")])
    _bot("old-bot", "hold", "hold", {}, [(old, "traded")])

    with patch.object(MarketDataService, "ensure_history", side_effect=AssertionError("descarga en ②")), \
            patch.object(MarketDataService, "sync_bars", side_effect=AssertionError("descarga en ②")):
        tasks.compute_regimes_and_signals()

    assert list(Signal.objects.values_list("instrument__symbol", flat=True)) == ["OLD"]
    assert get_or_create_today_pipeline().timings["signals"]["missing_history"] == ["NEW"]


@pytest.mark.django_db
def test_prewarm_downloads_missing_history_and_fundamentals_outside_the_window():
    new = _instrument("NEWCO", synced=False, asset_class="equity")
    unused = _instrument("UNUSED", synced=False)
    _bot("ranked", "ranked_allocation", "ranked_allocation", {"metrics": []}, [(new, "traded")])

    with _fake_provider(), \
            patch("apps.fundamentals.services.sync_instrument", return_value={"saved": 4}) as fundamentals:
        result = tasks.prewarm_market_data()

    new.refresh_from_db()
    unused.refresh_from_db()
    assert new.full_history_synced and Bar.objects.filter(instrument=new).exists()
    assert not unused.full_history_synced
    assert fundamentals.call_args.args[0] == new
    assert result.startswith("Prewarmed full history for 1 instruments")


@pytest.mark.django_db
def test_schedules_register_prewarm_at_1530():
    from django.core.management import call_command
    from django_celery_beat.models import PeriodicTask

    call_command("setup_botops_schedules", stdout=io.StringIO())
    prewarm = PeriodicTask.objects.get(task="prewarm_market_data")
    assert (prewarm.crontab.hour, prewarm.crontab.minute) == ("15", "30")
    assert tasks.prewarm_market_data.queue == "pipeline"
