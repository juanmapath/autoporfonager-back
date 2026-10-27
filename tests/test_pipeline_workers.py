import io
import re
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from django.conf import settings
from django.utils import timezone

from apps.instruments.models import Instrument
from apps.trading import tasks
from apps.trading.models import OrderIntent, RebalanceRun
from apps.trading.pipeline import get_or_create_today_pipeline
from tests.test_pipeline_safety import _paper_portfolio

ROOT = Path(settings.BASE_DIR)


def _worker_queues(text: str) -> dict:
    """{nombre_worker: {colas}} a partir de las líneas `celery ... worker -n <name> ... -Q a,b`."""
    workers = {}
    for line in text.splitlines():
        if "celery" not in line or " worker" not in line or line.strip().startswith("#"):
            continue
        queues = re.search(r"-Q ([\w,]+)", line).group(1)
        name = re.search(r"-n (\w+)@", line)
        workers[name.group(1) if name else queues] = set(queues.split(","))
    return workers


# ---------- Colas y workers (P3) ----------

def test_dispatch_runs_on_execution_queue_and_pipeline_on_pipeline_queue():
    assert tasks.dispatch_moc_and_manual_orders.queue == "execution"
    for task in (tasks.run_daily_trading_pipeline, tasks.orchestrate_portfolio_runs,
                 tasks.fallback_compute_signals, tasks.fallback_orchestrate):
        assert task.queue == "pipeline", task.name
    assert tasks.pipeline_watchdog.queue == "celery"
    assert tasks.pipeline_post_cutoff_check.queue == "celery"


@pytest.mark.parametrize("path", ["start.sh", "docker/compose.yml"])
def test_execution_worker_is_isolated_and_every_task_queue_is_consumed(path):
    workers = _worker_queues((ROOT / path).read_text(encoding="utf-8"))
    workers.pop("backtest", None)
    assert workers["exec"] == {"execution"}
    assert "execution" not in workers["main"]
    assert "pipeline" in workers["main"]

    consumed = set().union(*workers.values())
    task_queues = {
        obj.queue for obj in vars(tasks).values()
        if getattr(obj, "queue", None) and getattr(obj, "name", "") and hasattr(obj, "apply_async")
    }
    assert task_queues <= consumed, task_queues - consumed


@pytest.mark.django_db
def test_schedules_register_watchdogs():
    from django.core.management import call_command
    from django_celery_beat.models import PeriodicTask

    call_command("setup_botops_schedules", stdout=io.StringIO())
    watchdog = PeriodicTask.objects.get(task="pipeline_watchdog")
    post_cutoff = PeriodicTask.objects.get(task="pipeline_post_cutoff_check")
    assert (watchdog.crontab.hour, watchdog.crontab.minute) == ("15", "56")
    assert (post_cutoff.crontab.hour, post_cutoff.crontab.minute) == ("16", "01")


# ---------- Watchdog 15:56 (P6) ----------

@pytest.mark.django_db
def test_watchdog_is_silent_when_pipeline_is_done():
    run = get_or_create_today_pipeline()
    run.status = "done"
    run.save()
    with patch("apps.notifications.telegram.send_to_telegram") as telegram:
        assert tasks.pipeline_watchdog() == "Pipeline OK"
    telegram.assert_not_called()


@pytest.mark.django_db
def test_watchdog_alerts_with_step_status_when_pipeline_is_not_done():
    run = get_or_create_today_pipeline()
    run.status = "running"
    run.started_at = timezone.now() - timedelta(minutes=6)
    run.marketdata_status = "done"
    run.signals_status = "running"
    run.timings = {"marketdata": {"seconds": 95.2}}
    run.save()

    with patch("apps.notifications.telegram.send_to_telegram") as telegram:
        assert tasks.pipeline_watchdog() == "Alert sent"

    msg = telegram.call_args.args[0]
    assert "No terminó a tiempo" in msg
    assert "marketdata: done (95.2 s)" in msg
    assert "signals: running" in msg
    assert "orchestration: pending" in msg
    assert re.search(r"Transcurrido desde el inicio: 3\d\d s", msg)


@pytest.mark.django_db
def test_watchdog_alerts_when_pipeline_never_started():
    with patch("apps.notifications.telegram.send_to_telegram") as telegram:
        tasks.pipeline_watchdog()
    assert "No arrancó" in telegram.call_args.args[0]


@pytest.mark.django_db
def test_watchdog_escapes_markdown_in_errors():
    run = get_or_create_today_pipeline()
    run.status = "failed"
    run.errors = [{"step": "signals", "error": "KeyError: 'source_symbol'"}]
    run.save()
    with patch("apps.notifications.telegram.send_to_telegram") as telegram:
        tasks.pipeline_watchdog()
    assert "source\\_symbol" in telegram.call_args.args[0]


# ---------- Chequeo post-corte 16:01 (P6) ----------

def _intent(run, symbol, status="pending"):
    inst, _ = Instrument.objects.get_or_create(symbol=symbol, defaults={"name": symbol})
    return OrderIntent.objects.create(
        run=run, instrument=inst, net_delta_qty=Decimal("1"), idempotency_key=f"k_{run.id}_{symbol}", status=status,
    )


@pytest.mark.django_db
def test_post_cutoff_check_is_silent_when_everything_was_sent():
    portfolio = _paper_portfolio()
    run = RebalanceRun.objects.create(portfolio=portfolio, as_of=timezone.now(), status="done")
    _intent(run, "SPY", status="dispatched")
    # Intents pendientes de días anteriores no cuentan para el chequeo de hoy
    old = RebalanceRun.objects.create(portfolio=portfolio, as_of=timezone.now() - timedelta(days=3), status="executing")
    _intent(old, "QQQ")

    with patch("apps.notifications.telegram.send_to_telegram") as telegram:
        assert tasks.pipeline_post_cutoff_check() == "Post-cutoff OK"
    telegram.assert_not_called()


@pytest.mark.django_db
def test_post_cutoff_check_alerts_on_missed_cutoff_and_pending_intents():
    portfolio = _paper_portfolio()
    missed = RebalanceRun.objects.create(portfolio=portfolio, as_of=timezone.now(), status="missed_cutoff")
    executing = RebalanceRun.objects.create(portfolio=portfolio, as_of=timezone.now(), status="executing")
    _intent(executing, "SPY")

    with patch("apps.notifications.telegram.send_to_telegram") as telegram:
        result = tasks.pipeline_post_cutoff_check()

    assert result == "Alert sent: 1 missed_cutoff runs, 1 pending intents"
    msg = telegram.call_args.args[0]
    assert f"Run #{missed.id}" in msg and "missed\\_cutoff" in msg
    assert "SPY" in msg
