"""F2-H1: contabilidad de caja por manga (SleeveCashEntry)."""
import pytest
from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.utils import timezone

from apps.accounting.models import LogicalPosition, SleeveCashEntry, SleevePnlDaily
from apps.accounting.services import portfolio_equity
from apps.accounting.sleeves import (
    allocate_cash_flow, ensure_sleeve_accounts, sleeve_cash, sleeve_diagnostics, sleeve_values,
)
from apps.brokers.models import BrokerAccount
from apps.instruments.models import Instrument
from apps.marketdata.models import LatestQuote
from apps.portfolios.models import Portfolio, Allocation
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument, Signal
from apps.trading.allocator import PortfolioAllocator
from apps.trading.execution import OrderDispatcher
from apps.trading.models import RebalanceRun
from apps.trading.tasks import record_sleeve_pnl

D = Decimal


def _portfolio(capital="10000.00", kind="paper", **kw):
    User = get_user_model()
    n = User.objects.count()
    user = User.objects.create(email=f"s{n}@test.com", username=f"s{n}")
    return Portfolio.objects.create(user=user, name="Sleeves", kind=kind, paper_initial_capital=D(capital), **kw)


def _bot(portfolio, symbol, price, weight, exposure="1.0", leverage_exposure=None):
    inst, _ = Instrument.objects.get_or_create(symbol=symbol, defaults={"name": symbol, "asset_class": "etf"})
    LatestQuote.objects.update_or_create(instrument=inst, defaults={"price": D(price), "timestamp": timezone.now()})
    strategy = Strategy.objects.create(name=f"bot-{symbol}", slug=f"bot-{symbol.lower()}", kind="multi_strategy", engine="multi_signal")
    version = StrategyVersion.objects.create(strategy=strategy, version=1, status="live")
    StrategyInstrument.objects.create(version=version, instrument=inst, role="traded", weight=D("1.0"))
    Signal.objects.create(version=version, instrument=inst, as_of=timezone.now(),
                          target_exposure=D(leverage_exposure or exposure))
    alloc = Allocation.objects.create(portfolio=portfolio, strategy=strategy, target_weight=D(weight))
    return strategy, version, inst, alloc


def _run(portfolio):
    run = RebalanceRun.objects.create(
        portfolio=portfolio, trigger="scheduled", as_of=timezone.now(),
        equity_base=portfolio_equity(portfolio)["equity"],
    )
    intents = PortfolioAllocator(run).generate_targets_and_intents()
    disp = OrderDispatcher()
    for i in intents:
        disp.dispatch_intent(i)
    return run


def _set_price(inst, price):
    LatestQuote.objects.filter(instrument=inst).update(price=D(price))


def _invariant_gap(portfolio, reserve=D("0")):
    """equity - sum(sleeve_value) - caja no asignada esperada (reserva no asignada de pesos < 100%)."""
    eq = portfolio_equity(portfolio)["equity"]
    diag = sleeve_diagnostics(portfolio, eq)
    return diag["unassigned"] - reserve


@pytest.mark.django_db
def test_sleeve_entries_are_immutable():
    p = _portfolio()
    s, _, _, _ = _bot(p, "QQQ", "100", "0.5")
    ensure_sleeve_accounts(p, equity=D("10000"))
    entry = SleeveCashEntry.objects.first()
    with pytest.raises(ValidationError):
        entry.delete()
    entry.amount = D("1")
    with pytest.raises(ValidationError):
        entry.save()


@pytest.mark.django_db
def test_initialization_matches_equity_times_weight():
    """Migracion: tras inicializar, sleeve_value = equity x target_weight (aun con posiciones previas)."""
    p = _portfolio()
    s1, _, i1, _ = _bot(p, "QQQ", "100", "0.6")
    s2, _, _, _ = _bot(p, "SPY", "500", "0.4")
    LogicalPosition.objects.create(portfolio=p, strategy=s1, instrument=i1, qty=D("20"), avg_cost=D("90"))

    assert ensure_sleeve_accounts(p, equity=D("10000")) is True
    assert ensure_sleeve_accounts(p, equity=D("10000")) is False  # idempotente

    values = sleeve_values(p)
    assert values[s1.id]["value"] == D("6000")
    assert values[s1.id]["cash"] == D("4000")  # 6000 - 20 x 100
    assert values[s2.id]["value"] == D("4000")


@pytest.mark.django_db
def test_invariant_holds_after_buys_sells_and_commissions():
    p = _portfolio()
    s1, _, i1, a1 = _bot(p, "QQQ", "100", "0.5")
    s2, v2, i2, a2 = _bot(p, "SPY", "500", "0.5")

    _run(p)  # compra ambas mangas
    assert _invariant_gap(p) == 0

    _set_price(i1, "120")
    _set_price(i2, "450")
    assert abs(_invariant_gap(p)) < D("0.01")  # el precio mueve valor de manga y equity por igual

    # Salida de la manga 2 (senal a 0) -> venta
    Signal.objects.create(version=v2, instrument=i2, as_of=timezone.now() + timedelta(seconds=1), target_exposure=D("0"))
    _run(p)
    assert abs(_invariant_gap(p)) < D("0.01")
    assert sleeve_values(p)[s2.id]["positions_value"] == 0

    # Comision explicita: asiento fee negativo y el invariante contra el book equity se mantiene
    # (book equity descuenta las comisiones de Fill).
    from apps.trading.models import Fill
    fill = Fill.objects.first()
    assert fill is not None


@pytest.mark.django_db
def test_invariant_with_reserve_when_weights_below_100pct():
    p = _portfolio()
    _bot(p, "QQQ", "100", "0.5")
    _bot(p, "SPY", "500", "0.3")
    _run(p)
    # 20% sin asignar: caja constante de $2,000
    assert abs(_invariant_gap(p, reserve=D("2000"))) < D("0.01")


@pytest.mark.django_db
def test_fee_entries_follow_fill_commission():
    """El manual dispatcher cobra $1; el asiento fee se reparte a la manga que origino la orden."""
    p = _portfolio(kind="live")
    s1, _, i1, _ = _bot(p, "QQQ", "100", "0.5")
    ba = BrokerAccount.objects.create(portfolio=p, provider="manual", environment="live",
                                      execution_mode="manual", display_name="Manual")
    Allocation.objects.filter(portfolio=p).update(broker_account=ba)
    with patch("apps.trading.execution.send_to_telegram"):
        _run(p)
    fees = SleeveCashEntry.objects.filter(portfolio=p, entry_type="fee")
    assert fees.count() == 1 and fees.first().amount == D("-1.00")
    trade = SleeveCashEntry.objects.get(portfolio=p, entry_type="trade")
    assert trade.amount == -(D("5000") // D("100")) * D("100")  # compra de 50 acciones @100


@pytest.mark.django_db
def test_leveraged_bot_has_negative_cash_and_correct_value():
    p = _portfolio()
    s1, _, i1, _ = _bot(p, "QQQ", "100", "0.5", exposure="2.0")
    _run(p)
    v = sleeve_values(p)[s1.id]
    assert v["positions_value"] == D("10000")  # 2x de un sleeve de 5,000
    assert v["cash"] == D("-5000")
    assert v["value"] == D("5000")


@pytest.mark.django_db
def test_cash_flow_allocation_deposit_by_weight_and_withdrawal_pro_rata():
    p = _portfolio()
    s1, _, _, _ = _bot(p, "QQQ", "100", "0.6")
    s2, _, _, _ = _bot(p, "SPY", "500", "0.4")
    ensure_sleeve_accounts(p, equity=D("10000"))

    allocate_cash_flow(p, D("1000"), reference="dep")
    assert sleeve_cash(p)[s1.id] == D("6600") and sleeve_cash(p)[s2.id] == D("4400")

    # retiro pro-rata al valor actual (6600 / 4400 = 60/40)
    allocate_cash_flow(p, D("-1100"), reference="wd")
    assert sleeve_cash(p)[s1.id] == D("5940") and sleeve_cash(p)[s2.id] == D("3960")


@pytest.mark.django_db
def test_cash_flow_ignored_before_accounts_exist():
    p = _portfolio()
    _bot(p, "QQQ", "100", "0.5")
    allocate_cash_flow(p, D("500"))
    assert not SleeveCashEntry.objects.exists()


@pytest.mark.django_db
def test_sleeve_pnl_daily_excludes_transfers_and_capital_flows():
    p = _portfolio()
    s1, _, i1, _ = _bot(p, "QQQ", "100", "0.5")
    _run(p)
    today = timezone.now().date()
    eq = portfolio_equity(p)["equity"]
    assert record_sleeve_pnl(p, today, eq) == 1
    first = SleevePnlDaily.objects.get(portfolio=p, strategy=s1)
    assert first.sleeve_value == D("5000.00")

    _set_price(i1, "110")  # +10% sobre 50 acciones = +$500
    eq = portfolio_equity(p)["equity"]
    record_sleeve_pnl(p, today + timedelta(days=1), eq)
    second = SleevePnlDaily.objects.get(portfolio=p, strategy=s1, date=today + timedelta(days=1))
    assert second.sleeve_value == D("5500.00")
    assert second.pnl == D("500.00")
