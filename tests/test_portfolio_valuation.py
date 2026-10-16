import pytest
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.utils import timezone

from apps.accounting.models import LogicalPosition
from apps.accounting.services import open_positions, portfolio_equity
from apps.brokers.models import BrokerAccount
from apps.instruments.models import Instrument
from apps.marketdata.models import LatestQuote
from apps.portfolios.models import Portfolio, Allocation
from apps.strategies.models import Strategy


def _portfolio(capital="10000.00"):
    User = get_user_model()
    user = User.objects.create(email=f"v{User.objects.count()}@test.com", username=f"v{User.objects.count()}")
    return Portfolio.objects.create(user=user, name="Paper", kind="paper", paper_initial_capital=Decimal(capital))


def _position(portfolio, slug, symbol, qty, avg_cost, price, weight, realized="0"):
    strategy, _ = Strategy.objects.get_or_create(slug=slug, defaults={"name": slug})
    instrument, _ = Instrument.objects.get_or_create(symbol=symbol)
    LatestQuote.objects.update_or_create(
        instrument=instrument, defaults={"price": Decimal(price), "timestamp": timezone.now()},
    )
    Allocation.objects.update_or_create(
        portfolio=portfolio, strategy=strategy, defaults={"target_weight": Decimal(weight), "enabled": True},
    )
    return LogicalPosition.objects.create(
        portfolio=portfolio, strategy=strategy, instrument=instrument,
        qty=Decimal(qty), avg_cost=Decimal(avg_cost), realized_pnl=Decimal(realized),
    )


@pytest.mark.django_db
def test_open_positions_unrealized_pnl_and_excludes_closed():
    port = _portfolio()
    _position(port, "s1", "AAA", "10", "100", "110", "0.5")
    _position(port, "s2", "BBB", "0", "50", "60", "0.5", realized="25")

    result = open_positions(port, refresh=False)

    assert [p["instrument"] for p in result["positions"]] == ["AAA"]
    row = result["positions"][0]
    assert row["market_value"] == Decimal("1100")
    assert row["unrealized_pnl"] == Decimal("100")
    assert row["unrealized_pnl_pct"] == Decimal("10")
    assert result["summary"]["unrealized_pnl"] == Decimal("100")


@pytest.mark.django_db
def test_book_equity_includes_realized_and_unrealized():
    port = _portfolio("10000.00")
    _position(port, "s1", "AAA", "10", "100", "110", "0.5")
    _position(port, "s2", "BBB", "0", "50", "60", "0.5", realized="25")

    eq = portfolio_equity(port, refresh=False)

    assert eq["source"] == "book"
    assert eq["equity"] == Decimal("10125.00")


@pytest.mark.django_db
def test_broker_equity_wins_over_book():
    port = _portfolio("10000.00")
    BrokerAccount.objects.create(
        portfolio=port, provider="alpaca", environment="paper", display_name="A", managed_capital=Decimal("12345.67"),
    )

    eq = portfolio_equity(port, refresh=False)

    assert eq["source"] == "broker"
    assert eq["equity"] == Decimal("12345.67")


@pytest.mark.django_db
def test_sleeve_leverage_splits_own_and_financed_capital():
    port = _portfolio("10000.00")
    BrokerAccount.objects.create(
        portfolio=port, provider="alpaca", environment="paper", display_name="A", managed_capital=Decimal("10000"),
    )
    # Manga del 50% (5.000) con 10.000 invertidos -> 2x: 5.000 propios y 5.000 financiados
    _position(port, "lev", "TQQQ", "100", "100", "100", "0.5")

    result = open_positions(port, refresh=False)
    row = result["positions"][0]

    assert row["sleeve_leverage"] == Decimal("2")
    assert row["own_capital"] == Decimal("5000")
    assert row["financed"] == Decimal("5000")
    # A nivel cuenta la exposición (10.000) no supera el equity: no hay margen tomado
    assert result["summary"]["leverage"] == Decimal("1")
    assert result["summary"]["financed"] == Decimal("0")


@pytest.mark.django_db
def test_equity_history_returns_snapshots_and_benchmark_closes():
    from datetime import date, timedelta
    from apps.accounting.models import EquitySnapshot
    from apps.accounting.services import equity_history
    from apps.marketdata.models import Bar

    port = _portfolio("10000.00")
    d0 = date.today() - timedelta(days=3)
    EquitySnapshot.objects.create(portfolio=port, date=d0, total_equity=Decimal("10000"))
    spy = Instrument.objects.create(symbol="SPY", tradable=False)
    for i in range(14):
        d = d0 - timedelta(days=10) + timedelta(days=i)
        px = Decimal("500") + i
        Bar.objects.create(instrument=spy, date=d, open=px, high=px, low=px, close=px, adj_close=px)

    data = equity_history(port, ["SPY", "NOPE"])

    # Snapshot histórico + punto de hoy calculado en vivo
    from core.market_time import get_ny_now
    assert [p["date"] for p in data["portfolio"]] == [d0, get_ny_now().date()]
    assert list(data["benchmarks"]) == ["SPY"]
    assert data["benchmarks"]["SPY"]["series"][0]["date"] >= d0 - timedelta(days=7)


@pytest.mark.django_db
def test_orders_endpoint_includes_fill_price_and_date():
    from rest_framework.test import APIClient
    from apps.trading.models import Order, Fill

    port = _portfolio()
    inst = Instrument.objects.create(symbol="AAA")
    order = Order.objects.create(
        portfolio=port, instrument=inst, client_order_id="x1", side="buy", qty=Decimal("2"), status="filled",
    )
    filled_at = timezone.now()
    Fill.objects.create(order=order, fill_price=Decimal("10"), qty=Decimal("1"), filled_at=filled_at)
    Fill.objects.create(order=order, fill_price=Decimal("12"), qty=Decimal("1"), filled_at=filled_at)

    client = APIClient()
    client.force_authenticate(port.user)
    res = client.get(f"/api/v1/portfolios/{port.id}/orders/")

    assert res.status_code == 200
    assert res.data[0]["fill_price"] == "11.0000"
    assert res.data[0]["filled_at"] == filled_at


@pytest.mark.django_db
def test_log_returns_neutralize_deposits_and_withdrawals():
    import math
    from datetime import date, timedelta
    from apps.accounting.models import EquitySnapshot, LedgerEntry
    from apps.accounting.services import equity_series, time_weighted_return

    port = _portfolio("10000.00")
    from core.market_time import get_ny_now
    d = [get_ny_now().date() - timedelta(days=i) for i in (3, 2, 1, 0)]
    # +10% el día 1; depósito de 5.000 el día 2 sin rendimiento; retiro de 2.000 el día 3 con -1.000
    for day, eq in zip(d, ["10000", "11000", "16000", "13000"]):
        EquitySnapshot.objects.create(portfolio=port, date=day, total_equity=Decimal(eq))
    LedgerEntry.objects.create(portfolio=port, entry_type="deposit", amount=Decimal("5000"), effective_date=d[2])
    LedgerEntry.objects.create(portfolio=port, entry_type="withdrawal", amount=Decimal("-2000"), effective_date=d[3])

    points = equity_series(port)

    assert [round(p["log_return"], 6) for p in points] == [
        0.0, round(math.log(1.1), 6), 0.0, round(math.log(15000 / 16000), 6),
    ]
    assert points[-1]["contributed"] == Decimal("13000")
    assert round(time_weighted_return(points), 6) == round(1.1 * 15000 / 16000 - 1, 6)


@pytest.mark.django_db
def test_sync_alpaca_cash_flows_imports_once_and_skips_pre_link():
    from datetime import date, timedelta
    from unittest.mock import patch
    from apps.accounting.models import LedgerEntry
    from apps.brokers.models import BrokerCredential
    from apps.brokers.services import sync_alpaca_cash_flows

    port = _portfolio()
    cred = BrokerCredential(user=port.user, provider="alpaca", environment="paper")
    cred.set_secrets({"key_id": "k", "secret_key": "s"})
    cred.save()
    account = BrokerAccount.objects.create(
        portfolio=port, provider="alpaca", environment="paper", display_name="A", credential=cred,
    )
    from core.market_time import get_ny_now
    today = get_ny_now().date()
    activities = [
        {"id": "old", "activity_type": "CSD", "date": str(today - timedelta(days=30)), "net_amount": "999", "status": "executed"},
        {"id": "dep", "activity_type": "CSD", "date": str(today), "net_amount": "1500", "status": "executed"},
        {"id": "wd", "activity_type": "CSW", "date": str(today), "net_amount": "-500", "status": "executed"},
        {"id": "cx", "activity_type": "CSD", "date": str(today), "net_amount": "100", "status": "canceled"},
    ]
    with patch("apps.brokers.services.get_alpaca_cash_activities", return_value=activities):
        assert sync_alpaca_cash_flows(account) == 2
        assert sync_alpaca_cash_flows(account) == 0

    flows = {e.reference: (e.entry_type, e.amount) for e in LedgerEntry.objects.filter(portfolio=port)}
    assert flows == {"alpaca:dep": ("deposit", Decimal("1500")), "alpaca:wd": ("withdrawal", Decimal("-500"))}


@pytest.mark.django_db
def test_orchestration_sizes_on_synced_broker_equity():
    from unittest.mock import patch
    from apps.trading.models import RebalanceRun
    from apps.trading.tasks import orchestrate_portfolio_runs

    port = _portfolio("100000.00")
    BrokerAccount.objects.create(
        portfolio=port, provider="alpaca", environment="paper", display_name="A", managed_capital=Decimal("100"),
    )

    def fake_sync(account):
        account.managed_capital = Decimal("4321.00")
        account.save()
        return {"success": True}

    with patch("apps.trading.tasks.sync_alpaca_account", side_effect=fake_sync):
        orchestrate_portfolio_runs()

    run = RebalanceRun.objects.get(portfolio=port)
    assert run.equity_base == Decimal("4321.00")
    assert run.decisions["equity_source"] == "broker"
