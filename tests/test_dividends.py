"""F3-H1: dividendos en ledger (paper + Alpaca) y en _book_equity."""
import pytest
from datetime import date, datetime, timedelta, timezone as dt_tz
from decimal import Decimal
from unittest.mock import patch

from django.utils import timezone

from apps.accounting.dividends import accrue_paper_dividends, qty_held_on
from apps.accounting.models import LedgerEntry, LogicalPosition, SleeveCashEntry
from apps.accounting.services import (
    CASH_FLOW_TYPES, equity_series, net_cash_flows, portfolio_equity,
)
from apps.accounting.sleeves import ensure_sleeve_accounts, sleeve_values
from apps.brokers.models import BrokerAccount, BrokerCredential
from apps.brokers.services import sync_alpaca_dividends
from apps.marketdata.providers.yahoo import YahooProvider
from tests.test_sleeve_accounting import _portfolio, _bot, _run

D = Decimal
TODAY = timezone.now().date() + timedelta(days=1)  # ex-date posterior a las compras de los tests


class FakeProvider:
    def __init__(self, dividends):
        self.dividends = dividends

    def get_dividends(self, symbol, start_dt):
        return self.dividends.get(symbol, [])


def _held_portfolio():
    p = _portfolio()
    s1, _, i1, _ = _bot(p, "SGOV", "100", "0.5")
    s2, _, i2, _ = _bot(p, "SPY", "500", "0.5")
    _run(p)  # compra 50 SGOV y 10 SPY
    return p, s1, s2, i1, i2


def test_dividend_types_are_not_external_cash_flows():
    assert "dividend" not in CASH_FLOW_TYPES and "dividend_withholding" not in CASH_FLOW_TYPES


def test_yahoo_parses_dividend_events():
    raw = {"chart": {"result": [{"events": {"dividends": {
        "1": {"amount": 0.35, "date": int(datetime(2026, 11, 3, tzinfo=dt_tz.utc).timestamp())},
        "2": {"amount": 0.30, "date": int(datetime(2026, 10, 1, tzinfo=dt_tz.utc).timestamp())},
    }}}]}}
    out = YahooProvider._parse_dividends(raw)
    assert [d["ex_date"] for d in out] == [date(2026, 10, 1), date(2026, 11, 3)]
    assert YahooProvider._parse_dividends({"chart": {"result": [{}]}}) == []


@pytest.mark.django_db
def test_paper_dividend_enters_book_equity_the_right_sleeve_and_not_the_twr_flows():
    p, s1, s2, i1, i2 = _held_portfolio()
    before = portfolio_equity(p)["equity"]
    flows_before = net_cash_flows(p)

    provider = FakeProvider({"SGOV": [{"ex_date": TODAY, "amount": 0.30}]})
    result = accrue_paper_dividends(provider=provider, today=TODAY)
    assert result == {"created": 1, "errors": 0}

    entry = LedgerEntry.objects.get(portfolio=p, entry_type="dividend")
    assert entry.amount == D("15.00")  # 50 acciones x 0.30
    assert portfolio_equity(p)["equity"] == before + D("15.00")
    # No es un aporte externo
    assert net_cash_flows(p) == flows_before
    # Manga correcta (la que tiene SGOV), no la otra
    div = SleeveCashEntry.objects.get(portfolio=p, entry_type="dividend")
    assert div.strategy_id == s1.id and div.amount == D("15.00")
    assert sleeve_values(p)[s1.id]["value"] == D("5015")
    assert sleeve_values(p)[s2.id]["value"] == D("5000")

    # El flujo diario del TWR no incluye el dividendo: el retorno del dia lo cuenta como ganancia
    points = equity_series(p)
    assert all(pt["net_flow"] == 0 for pt in points)


@pytest.mark.django_db
def test_paper_dividend_is_idempotent():
    p, *_ = _held_portfolio()
    provider = FakeProvider({"SGOV": [{"ex_date": TODAY, "amount": 0.30}]})
    accrue_paper_dividends(provider=provider, today=TODAY)
    assert accrue_paper_dividends(provider=provider, today=TODAY)["created"] == 0
    assert LedgerEntry.objects.filter(portfolio=p, entry_type="dividend").count() == 1


@pytest.mark.django_db
def test_dividend_ignores_quantity_bought_on_or_after_ex_date_and_old_ex_dates():
    p, s1, _, i1, _ = _held_portfolio()
    pos = LogicalPosition.objects.get(portfolio=p, strategy=s1, instrument=i1)
    # La compra del _held_portfolio se hizo hoy (fecha real); con ex-date futura respecto a ella cuenta 0
    ex_after_buy = timezone.now().date()
    assert qty_held_on(p, pos, ex_after_buy) == 0
    assert qty_held_on(p, pos, ex_after_buy + timedelta(days=1)) == pos.qty

    provider = FakeProvider({"SGOV": [{"ex_date": TODAY - timedelta(days=30), "amount": 0.30}]})
    assert accrue_paper_dividends(provider=provider, today=TODAY)["created"] == 0


def _alpaca_portfolio():
    p = _portfolio(kind="live")
    s1, _, i1, _ = _bot(p, "SGOV", "100", "0.5")
    cred = BrokerCredential.objects.create(user=p.user, provider="alpaca", environment="live")
    cred.set_secrets({"key_id": "PK", "secret_key": "S"})
    cred.save()
    acct = BrokerAccount.objects.create(
        portfolio=p, credential=cred, provider="alpaca", environment="live", execution_mode="auto", display_name="Alp",
    )
    Allocation = type(p.allocations.first())
    Allocation.objects.update(broker_account=acct)
    LogicalPosition.objects.create(portfolio=p, broker_account=acct, strategy=s1, instrument=i1, qty=D("40"), avg_cost=D("100"))
    ensure_sleeve_accounts(p, equity=D("10000"))
    return p, acct, s1


@pytest.mark.django_db
def test_alpaca_div_and_divnra_are_imported_as_dividend_and_withholding():
    p, acct, s1 = _alpaca_portfolio()
    day = (timezone.now().date()).isoformat()
    activities = [
        {"id": "a1", "activity_type": "DIV", "symbol": "SGOV", "net_amount": "100.00", "date": day},
        {"id": "a2", "activity_type": "DIVNRA", "symbol": "SGOV", "net_amount": "-30.00", "date": day},
    ]
    with patch("apps.brokers.services.get_alpaca_cash_activities", return_value=activities), \
            patch("apps.brokers.services._alpaca_keys", return_value=("k", "s", "live")):
        assert sync_alpaca_dividends(acct) == 2
        assert sync_alpaca_dividends(acct) == 0  # idempotente

    div = LedgerEntry.objects.get(portfolio=p, entry_type="dividend")
    wh = LedgerEntry.objects.get(portfolio=p, entry_type="dividend_withholding")
    assert div.amount == D("100.00") and wh.amount == D("-30.00")
    assert net_cash_flows(p) == 0  # ninguno es aporte de capital
    # Neto atribuido a la manga dueña de la posicion
    assert sum(e.amount for e in SleeveCashEntry.objects.filter(portfolio=p, entry_type="dividend", strategy=s1)) == D("70.00")
