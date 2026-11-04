"""F2-H2/H3: rebalanceo en dos niveles (portafolio entre bots, bot entre activos)."""
import pytest
from contextlib import contextmanager
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import patch

from django.utils import timezone

from apps.accounting.models import SleeveCashEntry
from apps.accounting.sleeves import sleeve_values
from apps.portfolios.models import Allocation
from apps.strategies.models import AllocationSnapshot, Signal
from apps.trading.models import RebalanceRun
from apps.trading.rebalance_policy import bot_rebalance_config, is_bot_rebalance_date
from apps.trading.tasks import get_frozen_ranking
from core.market_time import is_last_business_day
from tests.test_sleeve_accounting import _portfolio, _bot, _run, _set_price

D = Decimal


@contextmanager
def dates(portfolio_date: bool, bot_date: bool):
    with patch("apps.trading.allocator.is_portfolio_rebalance_date", return_value=portfolio_date), \
            patch("apps.trading.allocator.is_bot_rebalance_date", return_value=bot_date):
        yield


def _mark_done(run):
    RebalanceRun.objects.filter(id=run.id).update(status="done", as_of=timezone.now())


def _transfers(portfolio):
    return SleeveCashEntry.objects.filter(portfolio=portfolio, entry_type="transfer")


def _two_bots(**portfolio_kw):
    p = _portfolio(**portfolio_kw)
    s1, v1, i1, a1 = _bot(p, "QQQ", "100", "0.5")
    s2, v2, i2, a2 = _bot(p, "SPY", "500", "0.5")
    with dates(False, False):
        _mark_done(_run(p))
    return p, (s1, v1, i1, a1), (s2, v2, i2, a2)


@pytest.mark.django_db
def test_winning_sleeve_keeps_its_gain_between_portfolio_dates():
    p, (s1, _, i1, _), (s2, _, _, _) = _two_bots(rebalance_frequency="quarterly")
    _set_price(i1, "200")  # el bot 1 duplica

    with dates(False, False):
        run = _run(p)

    qqq_target = run.targets.get(strategy=s1)
    assert qqq_target.sleeve_capital == D("10000")  # conserva la ganancia, no se recorta a equity x 50%
    assert qqq_target.delta_qty == 0
    assert not _transfers(p).exists()
    assert sleeve_values(p)[s2.id]["value"] == D("5000")


@pytest.mark.django_db
def test_portfolio_reset_only_on_date_and_when_band_is_exceeded():
    p, (s1, _, i1, _), (s2, _, i2, _) = _two_bots(rebalance_frequency="quarterly", rebalance_tolerance=D("0.10"))

    # Fecha de portafolio pero dentro de banda (+2% de desvio): sin transferencias
    _set_price(i1, "102")
    with dates(True, False):
        _mark_done(_run(p))
    assert not _transfers(p).exists()

    # Banda excedida pero no es fecha: sin transferencias
    _set_price(i1, "200")
    with dates(False, False):
        _mark_done(_run(p))
    assert not _transfers(p).exists()

    # Fecha + banda excedida: se reajustan todas a equity x peso, con suma de transferencias nula
    with dates(True, False):
        run = _run(p)
    assert run.decisions["portfolio_reset"] == "portfolio_band_exceeded"
    t = {e.strategy_id: e.amount for e in _transfers(p)}
    assert set(t) == {s1.id, s2.id}
    assert sum(t.values()) == 0 and t[s1.id] < 0 < t[s2.id]
    values = sleeve_values(p)
    assert values[s1.id]["value"] == values[s2.id]["value"]
    # la manga reajustada se redimensiona
    assert run.targets.get(strategy=s1).reason == "portfolio_reset"


@pytest.mark.django_db
def test_weight_change_resets_immediately_even_off_date():
    p, (s1, _, i1, a1), (s2, _, _, a2) = _two_bots(rebalance_frequency="quarterly")
    RebalanceRun.objects.update(as_of=timezone.now() - timedelta(minutes=1))
    a1.target_weight, a2.target_weight = D("0.7"), D("0.3")
    a1.save()
    a2.save()

    with dates(False, False):
        run = _run(p)
    assert run.decisions["portfolio_reset"] == "allocation_change"
    values = sleeve_values(p)
    assert values[s1.id]["value"] == D("7000") and values[s2.id]["value"] == D("3000")
    assert run.targets.get(strategy=s1).reason == "allocation_change"


@pytest.mark.django_db
def test_manual_rebalance_resets_immediately():
    p, (s1, _, i1, _), _ = _two_bots(rebalance_frequency="quarterly")
    _set_price(i1, "102")
    p.rebalance_pending = True
    p.save()
    with dates(False, False):
        run = _run(p)
    assert run.decisions["portfolio_reset"] == "manual_rebalance"


@pytest.mark.django_db
def test_monthly_bot_inside_quarterly_portfolio_resizes_assets_but_not_sleeve_capital():
    p, (s1, v1, i1, _), (s2, _, _, _) = _two_bots(rebalance_frequency="quarterly")
    v1.params = {"rebalance_frequency": "monthly", "rebalance_tolerance": "0.05", "rebalance_tolerance_mode": "relative"}
    v1.save()
    _set_price(i1, "115")  # sleeve 1 = 5750; dentro de la banda del portafolio (7%)
    Signal.objects.create(version=v1, instrument=i1, as_of=timezone.now() + timedelta(seconds=1), target_exposure=D("0.8"))

    def monthly_only(d, frequency="monthly"):
        return frequency == "monthly"

    with patch("apps.trading.rebalance_policy.is_last_business_day", side_effect=monthly_only):
        run = _run(p)

    t = run.targets.get(strategy=s1)
    assert t.reason == "bot_rebalance"
    assert t.delta_qty < 0  # redimensiona los activos del bot a 80% de su manga
    assert t.sleeve_capital == D("5750")  # el capital del bot no cambia hasta el trimestre
    assert not _transfers(p).exists()


@pytest.mark.django_db
def test_bot_config_inherits_portfolio_when_unset():
    p, (_, v1, _, _), _ = _two_bots(rebalance_frequency="weekly", rebalance_tolerance=D("0.2"))
    cfg = bot_rebalance_config(v1, p)
    assert (cfg.frequency, cfg.tolerance, cfg.mode) == ("weekly", D("0.2"), "relative")

    v1.params = {"rebalance_frequency": "never", "rebalance_tolerance": 0.25, "rebalance_tolerance_mode": "absolute"}
    cfg = bot_rebalance_config(v1, p)
    assert (cfg.frequency, cfg.tolerance, cfg.mode) == ("never", D("0.25"), "absolute")
    assert is_bot_rebalance_date(v1, p, date(2026, 10, 30)) is False  # never


@pytest.mark.django_db
def test_bot_config_treats_inherit_marker_as_unset():
    p, (_, v1, _, _), _ = _two_bots(rebalance_frequency="monthly")
    v1.params = {"rebalance_frequency": "inherit", "rebalance_tolerance": None}
    assert bot_rebalance_config(v1, p).frequency == "monthly"


@pytest.mark.django_db
def test_get_frozen_ranking_uses_the_bots_frequency_and_inherits_otherwise():
    p, (s1, v1, _, _), _ = _two_bots(rebalance_frequency="quarterly")
    AllocationSnapshot.objects.create(
        version=v1, as_of=timezone.now(), weights={},
        diagnostics={"AAPL": {"rank": 1}, "NVDA": {"rank": 2}},
    )
    today = date(2026, 10, 13)

    def monthly_only(d, frequency="monthly"):
        return frequency == "monthly"

    with patch("apps.trading.rebalance_policy.is_last_business_day", side_effect=monthly_only):
        # La version hereda "quarterly" del portafolio: no toca recalcular
        assert get_frozen_ranking(v1, today) == ["AAPL", "NVDA"]
        # Con frecuencia propia mensual si toca, aunque el portafolio sea trimestral
        v1.params = {"rebalance_frequency": "monthly"}
        assert get_frozen_ranking(v1, today) is None
        # Frecuencia propia "never": nunca (salvo rebalanceo pendiente)
        v1.params = {"rebalance_frequency": "never"}
        assert get_frozen_ranking(v1, today) == ["AAPL", "NVDA"]
        p.rebalance_pending = True
        p.save()
        assert get_frozen_ranking(v1, today) is None


@pytest.mark.parametrize("d,expected", [
    (date(2026, 2, 27), True),    # ultimo dia habil de febrero (28 es sabado)
    (date(2026, 2, 26), False),
    (date(2026, 5, 29), True),
    (date(2026, 8, 31), True),
    (date(2026, 11, 30), True),
    (date(2026, 11, 27), False),
    (date(2026, 3, 31), False),   # fin de trimestre calendario, no de "earnings"
    (date(2026, 12, 31), False),
    (date(2026, 9, 30), False),
])
def test_quarterly_earnings_only_last_business_day_of_feb_may_aug_nov(d, expected):
    assert is_last_business_day(d, frequency="quarterly_earnings") is expected


@pytest.mark.django_db
def test_serializer_stores_only_defined_bot_rebalance_params_and_exposes_sleeve_deviation():
    from apps.strategies.serializers import BotCreateSerializer
    from apps.portfolios.serializers import AllocationSerializer

    ser = BotCreateSerializer(data={"name": "Solo", "bot_type": "hold", "traded_symbol": "QQQ"})
    assert ser.is_valid(), ser.errors
    v = ser.save().versions.first()
    assert not any(k.startswith("rebalance_") for k in v.params)  # hereda del portafolio

    ser = BotCreateSerializer(data={
        "name": "Mensual", "bot_type": "hold", "traded_symbol": "SPY", "rebalance_frequency": "quarterly_earnings",
        "rebalance_tolerance": 0.25, "rebalance_tolerance_mode": "relative",
    })
    assert ser.is_valid(), ser.errors
    v = ser.save().versions.first()
    assert v.params["rebalance_frequency"] == "quarterly_earnings" and v.params["rebalance_tolerance"] == 0.25

    p, (s1, _, i1, a1), _ = _two_bots(rebalance_frequency="quarterly")
    _set_price(i1, "200")
    data = AllocationSerializer(a1).data
    assert data["sleeve_value"] == 10000.0
    assert data["weight_actual"] == round(10000 / 15000, 4)
    assert data["weight_deviation"] == round(10000 / 15000 - 0.5, 4)
