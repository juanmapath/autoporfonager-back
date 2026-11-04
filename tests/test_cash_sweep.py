"""F3-H2: cash sweep a T-bills (objetivo, exclusiones del allocator, orden de despacho)."""
import pytest
from datetime import timedelta
from decimal import Decimal

from django.utils import timezone

from apps.accounting.models import LogicalPosition
from apps.accounting.sleeves import release_disabled_sleeves, sleeve_values
from apps.instruments.models import Instrument
from apps.marketdata.models import LatestQuote
from apps.strategies.models import Signal
from apps.trading.cash_sweep import (
    SWEEP_KIND, get_sweep_strategy, order_intent_ids_sells_first, plan_sweep, sweep_instrument_ids,
)
from apps.trading.universe import pipeline_universe
from tests.test_sleeve_accounting import _portfolio, _bot, _run, _set_price

D = Decimal


def _sweep_portfolio(exposure="0", enabled=True, **kw):
    p = _portfolio(**kw)
    s, v, inst, alloc = _bot(p, "QQQ", "100", "0.5", exposure=exposure)
    sgov = Instrument.objects.create(symbol="SGOV", name="SGOV", asset_class="etf")
    LatestQuote.objects.create(instrument=sgov, price=D("100"), timestamp=timezone.now())
    p.cash_sweep_enabled = enabled
    p.save()
    return p, s, v, inst, sgov


def _signal(version, inst, exposure, seconds=1):
    Signal.objects.create(version=version, instrument=inst, as_of=timezone.now() + timedelta(seconds=seconds),
                          target_exposure=D(exposure))


def _sweep_targets(run):
    return list(run.targets.filter(strategy__kind=SWEEP_KIND))


def _sweep_qty(p):
    pos = LogicalPosition.objects.filter(portfolio=p, strategy__kind=SWEEP_KIND).first()
    return pos.qty if pos else D("0")


@pytest.mark.django_db
def test_sweep_disabled_creates_no_sweep_target_or_strategy():
    p, s, v, inst, _ = _sweep_portfolio(exposure="1", enabled=False)
    run = _run(p)
    assert _sweep_targets(run) == []
    assert not run.intents.filter(instrument__symbol="SGOV").exists()
    from apps.strategies.models import Strategy
    assert not Strategy.objects.filter(kind=SWEEP_KIND).exists()


@pytest.mark.django_db
def test_bots_without_signal_sweep_buys_equity_minus_buffer():
    p, *_ = _sweep_portfolio(exposure="0")
    run = _run(p)
    t, = _sweep_targets(run)
    assert t.reason == "cash_sweep_in"
    assert t.target_qty == D("98")  # 10,000 x (1 - 2%) / $100
    assert _sweep_qty(p) == D("98")


@pytest.mark.django_db
def test_bot_entry_sweep_sells_what_is_needed_and_sells_dispatch_first():
    p, s, v, inst, _ = _sweep_portfolio(exposure="0")
    _run(p)  # el sweep compra 98
    _signal(v, inst, "1.0")

    run = _run(p)
    sweep, = _sweep_targets(run)
    assert sweep.reason == "cash_sweep_out"
    assert sweep.delta_qty == D("-50")  # free = 10,000 - 5,000 - 200 = 4,800 -> 48 acciones
    assert _sweep_qty(p) == D("48")
    assert run.targets.get(strategy=s).delta_qty == D("50")  # la compra del bot sigue intacta

    # Orden de despacho: ventas (sweep primero) antes que compras. Recreamos los intents pendientes
    # del run para comprobar el orden (el _run ya los despacho).
    from apps.trading.models import OrderIntent
    run.intents.update(status="pending")
    ids = order_intent_ids_sells_first(run)
    intents = {i.id: i for i in OrderIntent.objects.filter(id__in=ids)}
    assert [intents[i].net_delta_qty < 0 for i in ids] == [True, False]
    assert intents[ids[0]].instrument.symbol == "SGOV"


@pytest.mark.django_db
def test_margin_usage_sets_sweep_to_zero():
    p, s, v, inst, _ = _sweep_portfolio(exposure="0")
    _run(p)
    _signal(v, inst, "2.5")  # 12,500 de exposicion con equity de 10,000
    run = _run(p)
    sweep, = _sweep_targets(run)
    assert sweep.target_qty == 0 and sweep.reason == "cash_sweep_out"
    assert _sweep_qty(p) == 0


@pytest.mark.django_db
def test_disabling_liquidates_the_sweep_on_next_run():
    p, *_ = _sweep_portfolio(exposure="0")
    _run(p)
    assert _sweep_qty(p) == D("98")
    p.cash_sweep_enabled = False
    p.save()
    run = _run(p)
    sweep, = _sweep_targets(run)
    assert sweep.reason == "cash_sweep_out" and sweep.delta_qty == D("-98")
    assert _sweep_qty(p) == 0


@pytest.mark.django_db
def test_disabled_allocation_sweep_does_not_liquidate_the_sweep_position():
    p, s, v, inst, _ = _sweep_portfolio(exposure="0")
    _run(p)
    run = _run(p)  # segundo run: el sweep no debe aparecer como "huerfano"
    assert not run.targets.filter(reason="liquidate_disabled_allocation", strategy__kind=SWEEP_KIND).exists()
    assert _sweep_qty(p) == D("98")


@pytest.mark.django_db
def test_sweep_cash_is_not_released_as_a_disabled_sleeve():
    p, s, v, inst, _ = _sweep_portfolio(exposure="0")
    _run(p)
    _set_price(Instrument.objects.get(symbol="SGOV"), "101")
    _signal(v, inst, "1.0")
    _run(p)
    sweep = get_sweep_strategy()
    before = sleeve_values(p)[sweep.id]["cash"]
    release_disabled_sleeves(p)
    assert sleeve_values(p)[sweep.id]["cash"] == before


@pytest.mark.django_db
def test_sweep_instrument_enters_the_marketdata_universe_only_when_enabled():
    p, *_ = _sweep_portfolio(exposure="0", enabled=False)
    sgov = Instrument.objects.get(symbol="SGOV")
    assert sgov.id not in set(pipeline_universe().values_list("id", flat=True))
    p.cash_sweep_enabled = True
    p.save()
    assert sgov.id in sweep_instrument_ids()
    assert sgov.id in set(pipeline_universe().values_list("id", flat=True))


@pytest.mark.parametrize("kwargs,expected_reason,expected_delta", [
    # ajuste chico (< min_trade) sin necesidad de caja -> no opera
    (dict(actual_qty=D("98"), long_mv=D("0"), equity=D("10040"), bot_buys_value=D("0")), "cash_sweep_hold", D("0")),
    # mismo ajuste pero un bot necesita caja y hay que vender -> opera aunque sea poco
    (dict(actual_qty=D("98"), long_mv=D("100"), equity=D("10040"), bot_buys_value=D("100")), "cash_sweep_out", None),
    # ajuste grande -> opera
    (dict(actual_qty=D("0"), long_mv=D("0"), equity=D("10000"), bot_buys_value=D("0")), "cash_sweep_in", D("98")),
])
def test_plan_sweep_min_trade_and_funding_rules(kwargs, expected_reason, expected_delta):
    plan = plan_sweep(
        enabled=True, buffer_pct=D("0.02"), min_trade_usd=D("500"), price=D("100"), **kwargs,
    )
    assert plan.reason == expected_reason
    if expected_delta is not None:
        assert plan.delta_qty == expected_delta
    else:
        assert plan.delta_qty < 0


def test_plan_sweep_disabled_or_leveraged_liquidates_even_below_min_trade():
    for kw in (dict(enabled=False, long_mv=D("0")), dict(enabled=True, long_mv=D("11000"))):
        plan = plan_sweep(equity=D("10000"), buffer_pct=D("0.02"), min_trade_usd=D("500"),
                          price=D("100"), actual_qty=D("1"), bot_buys_value=D("0"), **kw)
        assert plan.target_qty == 0 and plan.delta_qty == D("-1") and plan.reason == "cash_sweep_out"
