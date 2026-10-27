import pytest
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model

from apps.instruments.models import Instrument
from apps.marketdata.providers.fake import FakeProvider
from apps.portfolios.models import Portfolio, Allocation
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument, Signal
from apps.trading.models import DailyPipelineRun, RebalanceRun
from apps.trading.pipeline import get_or_create_today_pipeline, pipeline_step
from apps.trading.tasks import (
    compute_regimes_and_signals,
    dispatch_moc_and_manual_orders,
    marketdata_refresh,
    run_daily_trading_pipeline,
)
from core.market_time import get_ny_now


# ---------- helpers ----------

def _setup_bot_and_portfolio():
    qqq = Instrument.objects.create(symbol="QQQ", name="QQQ", asset_class="etf")
    strategy = Strategy.objects.create(name="hold-qqq", slug="hold-qqq", kind="hold", engine="hold")
    version = StrategyVersion.objects.create(strategy=strategy, version=1, status="live")
    StrategyInstrument.objects.create(version=version, instrument=qqq, role="traded", weight=Decimal("1.0"))

    User = get_user_model()
    user = User.objects.create(email="p@test.com", username="p")
    portfolio = Portfolio.objects.create(user=user, name="Paper", kind="paper", paper_initial_capital=Decimal("10000"))
    Allocation.objects.create(portfolio=portfolio, strategy=strategy, version=version, target_weight=Decimal("0.5"))
    return portfolio, strategy, version, qqq


def _fake_provider():
    return patch("apps.marketdata.services.YahooProvider", lambda *a, **k: FakeProvider())


# ---------- H0: instrumentación ----------

@pytest.mark.django_db
def test_pipeline_run_records_each_step_with_timings():
    """Una corrida del pipeline crea el registro del día con los 3 pasos y sus duraciones."""
    _setup_bot_and_portfolio()

    with _fake_provider():
        run_daily_trading_pipeline()

    run = DailyPipelineRun.objects.get(trade_date=get_ny_now().date())
    assert run.status == "done"
    assert (run.marketdata_status, run.signals_status, run.orchestration_status) == ("done", "done", "done")
    assert run.started_at and run.finished_at and run.finished_at >= run.started_at
    for step in ("marketdata", "signals", "orchestration"):
        assert run.timings[step]["seconds"] >= 0
    assert run.timings["marketdata"]["instruments"] == 1
    assert run.timings["signals"]["versions"] == 1
    assert set(run.timings["signals"]["per_version"]) == {"hold-qqq@v1"}
    assert run.timings["orchestration"]["runs"] == 1
    assert run.errors == []
    assert Signal.objects.count() == 1
    assert RebalanceRun.objects.count() == 1


@pytest.mark.django_db
def test_failed_step_is_recorded_and_next_steps_still_run():
    _setup_bot_and_portfolio()

    with _fake_provider(), \
            patch("apps.trading.tasks.StrategyVersion.objects.filter", side_effect=RuntimeError("boom")):
        run_daily_trading_pipeline()

    run = DailyPipelineRun.objects.get(trade_date=get_ny_now().date())
    assert run.status == "failed"
    assert run.marketdata_status == "done"
    assert run.signals_status == "failed"
    assert run.orchestration_status == "done"
    assert run.errors[0]["step"] == "signals"
    assert "boom" in run.errors[0]["error"]
    assert RebalanceRun.objects.count() == 1


@pytest.mark.django_db
def test_signals_step_records_duration_per_bot_when_called_directly():
    _setup_bot_and_portfolio()
    with _fake_provider():
        marketdata_refresh()
        compute_regimes_and_signals()

    run = get_or_create_today_pipeline()
    assert run.signals_status == "done"
    assert run.timings["signals"]["per_version"]["hold-qqq@v1"] >= 0


@pytest.mark.django_db
def test_dispatch_records_duration_and_order_count():
    dispatch_moc_and_manual_orders()
    run = get_or_create_today_pipeline()
    assert run.timings["dispatch"]["orders"] == 0
    assert run.timings["dispatch"]["seconds"] >= 0


@pytest.mark.django_db
def test_step_becoming_done_after_failure_completes_the_day():
    """Si un paso fallido se completa después (fallback), el día queda done."""
    run = get_or_create_today_pipeline()
    run.marketdata_status = run.orchestration_status = "done"
    run.signals_status = "failed"
    run.status = "failed"
    run.save()

    with pipeline_step("signals"):
        pass

    run.refresh_from_db()
    assert run.signals_status == "done"
    assert run.status == "done"
