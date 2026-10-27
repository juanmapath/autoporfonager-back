import io
import threading
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.db import connection
from django.utils import timezone

from apps.accounting.models import EquitySnapshot
from apps.instruments.models import Instrument
from apps.portfolios.models import Portfolio
from apps.strategies.models import Signal
from apps.trading.guards import NOT_TRADING_DAY
from apps.trading.models import DailyPipelineRun, OrderIntent, Order, RebalanceRun
from apps.trading.pipeline import get_or_create_today_pipeline
from apps.trading.execution import OrderDispatcher
from apps.trading import tasks
from core.market_time import NY_TZ, get_ny_now
from tests.test_pipeline_instrumentation import _fake_provider, _setup_bot_and_portfolio


# ---------- helpers ----------

def _paper_portfolio(name="P2"):
    User = get_user_model()
    user = User.objects.create(email=f"{name}@test.com", username=name)
    return Portfolio.objects.create(user=user, name=name, kind="paper", paper_initial_capital=Decimal("5000"))


def _executing_run_with_intent(portfolio, symbol="SPY"):
    inst = Instrument.objects.create(symbol=symbol, name=symbol, asset_class="etf")
    run = RebalanceRun.objects.create(portfolio=portfolio, trigger="scheduled", as_of=timezone.now(), status="executing")
    intent = OrderIntent.objects.create(
        run=run, instrument=inst, net_delta_qty=Decimal("2"), idempotency_key=f"run_{run.id}_{symbol}", status="pending",
    )
    return run, intent


@contextmanager
def _lock_not_acquired(*args, **kwargs):
    yield False


def _set_steps(**statuses):
    run = get_or_create_today_pipeline()
    for step, status in statuses.items():
        setattr(run, f"{step}_status", status)
    run.save()
    return run


# ---------- P1: idempotencia fuerte del rebalanceo ----------

@pytest.mark.django_db
def test_orchestrate_twice_creates_one_run_per_portfolio_and_day():
    portfolio, *_ = _setup_bot_and_portfolio()
    other = _paper_portfolio()

    tasks.orchestrate_portfolio_runs()
    # Segunda llamada: el paso ya está done → se omite
    assert tasks.orchestrate_portfolio_runs().startswith("Skipped")
    # Aun forzando el paso, el re-chequeo por portafolio impide un segundo run
    tasks.orchestrate_portfolio_runs(force=True)

    assert RebalanceRun.objects.filter(portfolio=portfolio).count() == 1
    assert RebalanceRun.objects.filter(portfolio=other).count() == 1


@pytest.mark.django_db
def test_run_is_claimed_inside_atomic_with_select_for_update_and_recheck():
    """
    El portafolio se bloquea con select_for_update dentro de transaction.atomic() y el chequeo de
    runs existentes se hace DESPUÉS del bloqueo: si otro proceso creó el run mientras esperábamos
    el lock, no se crea un segundo.
    """
    portfolio, *_ = _setup_bot_and_portfolio()
    baseline_depth = len(connection.atomic_blocks)
    depth_at_lock = []
    real_select_for_update = Portfolio.objects.select_for_update

    def concurrent_winner_then_lock():
        depth_at_lock.append(len(connection.atomic_blocks))
        RebalanceRun.objects.create(portfolio=portfolio, trigger="scheduled", as_of=timezone.now(), status="planned")
        return real_select_for_update()

    with patch.object(Portfolio.objects, "select_for_update", side_effect=concurrent_winner_then_lock):
        assert tasks.claim_scheduled_run(portfolio, get_ny_now().date()) is None

    assert depth_at_lock == [baseline_depth + 1]
    assert RebalanceRun.objects.filter(portfolio=portfolio).count() == 1


@pytest.mark.django_db
def test_portfolio_is_skipped_when_redis_lock_is_not_acquired():
    _setup_bot_and_portfolio()
    with patch("apps.trading.tasks.redis_lock", _lock_not_acquired):
        tasks.orchestrate_portfolio_runs()
    assert not RebalanceRun.objects.exists()


@pytest.mark.django_db
def test_step_is_skipped_when_step_lock_is_held_by_another_process():
    _setup_bot_and_portfolio()
    with patch("core.locks.redis_lock", _lock_not_acquired):
        assert "already running" in tasks.compute_regimes_and_signals()
        assert "already running" in tasks.orchestrate_portfolio_runs()
    assert not Signal.objects.exists()
    assert not RebalanceRun.objects.exists()


@pytest.mark.django_db
def test_failed_portfolio_marks_run_failed_and_others_continue():
    portfolio, *_ = _setup_bot_and_portfolio()
    other = _paper_portfolio()
    real_equity = tasks.portfolio_equity

    def equity(p):
        if p.pk == portfolio.pk:
            raise RuntimeError("broker down")
        return real_equity(p)

    with patch("apps.trading.tasks.portfolio_equity", side_effect=equity):
        with pytest.raises(RuntimeError):
            tasks.orchestrate_portfolio_runs()

    assert RebalanceRun.objects.get(portfolio=portfolio).status == "failed"
    assert RebalanceRun.objects.get(portfolio=other).status == "done"
    assert get_or_create_today_pipeline().orchestration_status == "failed"

    # El fallback reintenta solo el portafolio fallido
    tasks.fallback_orchestrate()
    assert RebalanceRun.objects.filter(portfolio=portfolio, status="done").count() == 1
    assert RebalanceRun.objects.filter(portfolio=other).count() == 1


@pytest.mark.postgres
@pytest.mark.django_db(transaction=True)
def test_concurrent_claims_create_a_single_run_on_postgres():
    if connection.vendor != "postgresql":
        pytest.skip("select_for_update solo serializa de verdad en Postgres")
    portfolio = _paper_portfolio()
    barrier = threading.Barrier(2)
    results = []

    def worker():
        from django.db import connections
        barrier.wait()
        try:
            results.append(tasks.claim_scheduled_run(portfolio, get_ny_now().date()))
        finally:
            connections.close_all()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len([r for r in results if r is not None]) == 1
    assert RebalanceRun.objects.filter(portfolio=portfolio).count() == 1


# ---------- P4: días no hábiles ----------

@pytest.mark.django_db
def test_no_trading_task_writes_or_sends_orders_on_non_trading_day():
    portfolio, *_ = _setup_bot_and_portfolio()
    run, intent = _executing_run_with_intent(portfolio)
    _set_steps(marketdata="pending")

    trading_tasks = [
        tasks.run_daily_trading_pipeline, tasks.compute_regimes_and_signals, tasks.orchestrate_portfolio_runs,
        tasks.dispatch_moc_and_manual_orders, tasks.reconcile_broker_fills, tasks.reconcile_positions_vs_broker,
        tasks.accounting_daily_snapshot, tasks.fallback_compute_signals, tasks.fallback_orchestrate,
    ]
    with patch("apps.trading.guards.is_trading_day", return_value=False), _fake_provider(), \
            patch.object(OrderDispatcher, "dispatch_intent") as dispatch:
        for task in trading_tasks:
            assert task() == NOT_TRADING_DAY, task.name

    dispatch.assert_not_called()
    assert not Signal.objects.exists()
    assert RebalanceRun.objects.count() == 1
    assert not Order.objects.exists()
    assert not EquitySnapshot.objects.exists()
    intent.refresh_from_db()
    assert intent.status == "pending"
    pipeline = DailyPipelineRun.objects.get()
    assert pipeline.status == "skipped"
    assert pipeline.marketdata_status == pipeline.signals_status == pipeline.orchestration_status == "skipped"


@pytest.mark.django_db
def test_alpaca_dispatch_is_rejected_when_there_is_no_decision_window():
    from apps.brokers.models import BrokerAccount, BrokerCredential

    portfolio = _paper_portfolio()
    cred = BrokerCredential.objects.create(user=portfolio.user, provider="alpaca", environment="paper")
    cred.set_secrets({"key_id": "PKTEST", "secret_key": "SECRET"})
    cred.save()
    account = BrokerAccount.objects.create(
        portfolio=portfolio, credential=cred, provider="alpaca", environment="paper",
        execution_mode="auto", display_name="Alpaca Paper",
    )
    run, intent = _executing_run_with_intent(portfolio)
    intent.broker_account = account
    intent.save()

    saturday_1557 = datetime(2026, 10, 24, 15, 57, tzinfo=NY_TZ)
    with patch("apps.trading.execution.get_ny_now", return_value=saturday_1557), \
            patch("apps.trading.execution.send_to_telegram") as telegram, \
            patch("alpaca.trading.client.TradingClient") as client_cls:
        with pytest.raises(RuntimeError):
            OrderDispatcher().dispatch_intent(intent)

    client_cls.return_value.submit_order.assert_not_called()
    telegram.assert_called_once()
    intent.refresh_from_db()
    run.refresh_from_db()
    assert intent.status == "cancelled"
    assert run.status == "missed_cutoff"
    assert not Order.objects.exists()


# ---------- P2: fallbacks condicionales ----------

@pytest.mark.django_db
@pytest.mark.parametrize("status", ["done", "running"])
def test_fallbacks_skip_when_step_is_done_or_running(status):
    _setup_bot_and_portfolio()
    _set_steps(signals=status, orchestration=status)

    assert tasks.fallback_compute_signals() == f"Skipped: signals already {status} today"
    assert tasks.fallback_orchestrate() == f"Skipped: orchestration already {status} today"
    assert not Signal.objects.exists()
    assert not RebalanceRun.objects.exists()


@pytest.mark.django_db
def test_fallbacks_run_when_step_failed_and_warn_about_stale_signals():
    _setup_bot_and_portfolio()
    _set_steps(signals="failed", orchestration="failed")

    with _fake_provider():
        tasks.marketdata_refresh()
        assert tasks.fallback_compute_signals() == "Computed 1 signals"
    assert get_or_create_today_pipeline().signals_status == "done"

    _set_steps(signals="failed")
    with patch("apps.notifications.telegram.send_to_telegram") as telegram:
        assert tasks.fallback_orchestrate() == "Orchestrated 1 portfolio runs"
    assert "señales no actualizadas" in telegram.call_args.args[0]
    assert get_or_create_today_pipeline().orchestration_status == "done"


@pytest.mark.django_db
def test_fallback_orchestrate_does_not_warn_when_signals_are_fresh():
    _setup_bot_and_portfolio()
    _set_steps(signals="done", orchestration="pending")
    with patch("apps.notifications.telegram.send_to_telegram") as telegram:
        tasks.fallback_orchestrate()
    telegram.assert_not_called()
    assert RebalanceRun.objects.count() == 1


# ---------- Despacho concurrente ----------

@pytest.mark.django_db
def test_intent_already_dispatched_by_another_process_is_not_sent_again():
    portfolio = _paper_portfolio()
    run, intent = _executing_run_with_intent(portfolio)
    stale_ids = [intent.id]  # lista leída por este proceso antes de que el otro despachara

    # Otro despacho concurrente lo envía primero
    OrderIntent.objects.filter(pk=intent.pk).update(status="dispatched")

    with patch.object(OrderDispatcher, "dispatch_intent") as dispatch:
        assert tasks.dispatch_pending_intent(OrderDispatcher(), stale_ids[0]) is False
    dispatch.assert_not_called()


@pytest.mark.django_db
def test_dispatching_twice_sends_each_intent_once():
    portfolio = _paper_portfolio()
    _executing_run_with_intent(portfolio, "SPY")
    sent = []

    def fake_dispatch(self, intent):
        sent.append(intent.id)
        intent.status = "dispatched"
        intent.save(update_fields=["status"])

    with patch.object(OrderDispatcher, "dispatch_intent", fake_dispatch):
        tasks.dispatch_moc_and_manual_orders()
        # El run sigue `executing` (p. ej. un despacho que se solapa con otro): no se reenvía nada
        RebalanceRun.objects.update(status="executing")
        tasks.dispatch_moc_and_manual_orders()

    assert len(sent) == 1


@pytest.mark.django_db
def test_cutoff_rejection_inside_dispatch_transaction_is_persisted():
    """La excepción de corte se captura dentro de la transacción: cancelled/missed_cutoff no se pierden."""
    portfolio = _paper_portfolio()
    run, intent = _executing_run_with_intent(portfolio)

    def reject(self, intent):
        intent.status = "cancelled"
        intent.save(update_fields=["status"])
        intent.run.status = "missed_cutoff"
        intent.run.save(update_fields=["status"])
        raise RuntimeError("cutoff")

    with patch.object(OrderDispatcher, "dispatch_intent", reject):
        assert tasks.dispatch_pending_intent(OrderDispatcher(), intent.id) is False

    intent.refresh_from_db()
    run.refresh_from_db()
    assert intent.status == "cancelled"
    assert run.status == "missed_cutoff"


# ---------- redis_lock ----------

def test_redis_lock_propagates_errors_raised_inside_the_block():
    from core.locks import redis_lock

    with pytest.raises(ValueError):
        with redis_lock("test:key") as acquired:
            assert acquired is True
            raise ValueError("inside")


# ---------- Schedule ----------

@pytest.mark.django_db
def test_schedules_replace_unconditional_fallbacks_with_conditional_ones():
    from django.core.management import call_command
    from django_celery_beat.models import CrontabSchedule, PeriodicTask

    cron = CrontabSchedule.objects.create(minute="52", hour="15", day_of_week="1-5")
    PeriodicTask.objects.create(
        name="[2/6] Fallback: Compute Quantitative Signals", task="compute_regimes_and_signals", crontab=cron,
    )
    PeriodicTask.objects.create(
        name="[3/6] Fallback: Portfolio Rebalance & Allocation Orchestration", task="orchestrate_portfolio_runs", crontab=cron,
    )
    call_command("setup_botops_schedules", stdout=io.StringIO())

    tasks_by_name = dict(PeriodicTask.objects.values_list("name", "task"))
    assert "compute_regimes_and_signals" not in tasks_by_name.values()
    assert "orchestrate_portfolio_runs" not in tasks_by_name.values()
    assert tasks_by_name["[2/6] Conditional Fallback: Compute Quantitative Signals"] == "fallback_compute_signals"
    assert tasks_by_name["[3/6] Conditional Fallback: Portfolio Rebalance & Allocation Orchestration"] == "fallback_orchestrate"
