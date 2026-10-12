import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.utils import timezone

from apps.accounting.models import LogicalPosition
from apps.instruments.models import Instrument
from apps.marketdata.models import LatestQuote
from apps.portfolios.models import Portfolio, Allocation
from apps.strategies.backtest import BacktestEngine
from apps.strategies.engines.multi_signal import MultiSignalEngine
from apps.strategies.engines.single_signal import SingleSignalEngine
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument, Signal
from apps.strategies.signals.sizing import current_trade_leverage, pulses_to_long_state
from apps.trading.allocator import PortfolioAllocator
from apps.trading.execution import OrderDispatcher
from apps.trading.models import RebalanceRun, Order, FillAllocation
from apps.trading.tasks import diff_positions
from core.market_time import NY_TZ


# ---------- helpers ----------

def _make_portfolio(capital="4000.00"):
    User = get_user_model()
    user = User.objects.create(email=f"u{User.objects.count()}@test.com", username=f"u{User.objects.count()}")
    return Portfolio.objects.create(user=user, name="Paper", kind="paper", paper_initial_capital=Decimal(capital))


def _make_strategy(slug, instrument, kind="multi_strategy", engine="multi_signal"):
    strategy = Strategy.objects.create(name=slug, slug=slug, kind=kind, engine=engine)
    version = StrategyVersion.objects.create(strategy=strategy, version=1, status="live")
    StrategyInstrument.objects.create(version=version, instrument=instrument, role="traded", weight=Decimal("1.0"))
    return strategy, version


def _set_signal(version, instrument, exposure):
    Signal.objects.create(
        version=version, instrument=instrument, as_of=timezone.now(), target_exposure=Decimal(exposure),
    )


def _set_quote(instrument, price):
    LatestQuote.objects.update_or_create(
        instrument=instrument, defaults={"price": Decimal(price), "timestamp": timezone.now()},
    )


def _run_and_dispatch(portfolio):
    run = RebalanceRun.objects.create(
        portfolio=portfolio, trigger="scheduled", as_of=timezone.now(), equity_base=portfolio.paper_initial_capital,
    )
    intents = PortfolioAllocator(run).generate_targets_and_intents()
    dispatcher = OrderDispatcher()
    for intent in intents:
        dispatcher.dispatch_intent(intent)
    return run, intents


# ---------- 1. Atribución de fills ----------

@pytest.mark.django_db
def test_fill_is_attributed_to_originating_strategy_and_not_rebought():
    """
    Regresión del size duplicado: el fill debe quedar en la estrategia que generó la orden
    (no en la primera asignación del portafolio) y el siguiente run no debe volver a comprar.
    """
    portfolio = _make_portfolio()
    naut = Instrument.objects.create(symbol="NAUT", name="NAUT", asset_class="equity")
    qqq = Instrument.objects.create(symbol="QQQ", name="QQQ", asset_class="etf")
    _set_quote(naut, "20.00")
    _set_quote(qqq, "750.00")

    # La primera asignación habilitada NO es la que opera QQQ
    naut_strat, naut_ver = _make_strategy("naut-zdema", naut, kind="one_strategy", engine="single_signal")
    qqq_strat, qqq_ver = _make_strategy("qqq11", qqq)
    Allocation.objects.create(portfolio=portfolio, strategy=naut_strat, target_weight=Decimal("0.5"))
    Allocation.objects.create(portfolio=portfolio, strategy=qqq_strat, target_weight=Decimal("0.5"))
    _set_signal(naut_ver, naut, "0.0")
    _set_signal(qqq_ver, qqq, "1.35")

    _, intents = _run_and_dispatch(portfolio)
    assert len(intents) == 1

    expected_qty = Decimal("3.6")  # 4000 * 0.5 * 1.35 / 750
    positions = LogicalPosition.objects.filter(portfolio=portfolio, qty__gt=0)
    assert positions.count() == 1
    pos = positions.first()
    assert pos.strategy == qqq_strat
    assert pos.instrument == qqq
    assert pos.qty == expected_qty
    assert FillAllocation.objects.get().strategy == qqq_strat

    # Segundo día con la misma señal: no debe generarse ninguna orden nueva
    _, intents_day2 = _run_and_dispatch(portfolio)
    assert intents_day2 == []
    assert Order.objects.filter(portfolio=portfolio).count() == 1


@pytest.mark.django_db
def test_netted_fill_is_split_across_strategies_by_target_delta():
    """Dos estrategias sobre el mismo activo: la orden neta se reparte según el delta de cada una."""
    portfolio = _make_portfolio("10000.00")
    qqq = Instrument.objects.create(symbol="QQQ", name="QQQ", asset_class="etf")
    _set_quote(qqq, "500.00")

    strat_a, ver_a = _make_strategy("qqq-a", qqq)
    strat_b, ver_b = _make_strategy("qqq-b", qqq)
    Allocation.objects.create(portfolio=portfolio, strategy=strat_a, target_weight=Decimal("0.5"))
    Allocation.objects.create(portfolio=portfolio, strategy=strat_b, target_weight=Decimal("0.5"))
    LogicalPosition.objects.create(
        portfolio=portfolio, strategy=strat_b, instrument=qqq, qty=Decimal("4.0"), avg_cost=Decimal("450.00"),
    )
    _set_signal(ver_a, qqq, "1.0")   # A compra 10
    _set_signal(ver_b, qqq, "0.0")   # B vende 4

    _, intents = _run_and_dispatch(portfolio)
    assert len(intents) == 1
    assert intents[0].net_delta_qty == Decimal("6.0")

    pos_a = LogicalPosition.objects.get(portfolio=portfolio, strategy=strat_a, instrument=qqq)
    pos_b = LogicalPosition.objects.get(portfolio=portfolio, strategy=strat_b, instrument=qqq)
    assert pos_a.qty == Decimal("10.0")
    assert pos_b.qty == Decimal("0.0")
    assert pos_b.realized_pnl == Decimal("200.00")  # (500 - 450) * 4


# ---------- 2. Posiciones de asignaciones deshabilitadas ----------

@pytest.mark.django_db
def test_allocator_liquidates_positions_of_disabled_allocations():
    portfolio = _make_portfolio()
    goog = Instrument.objects.create(symbol="GOOG", name="GOOG", asset_class="equity")
    _set_quote(goog, "200.00")
    strat, _ = _make_strategy("goog-zdema", goog, kind="one_strategy", engine="single_signal")
    Allocation.objects.create(portfolio=portfolio, strategy=strat, target_weight=Decimal("0"), enabled=False)
    LogicalPosition.objects.create(
        portfolio=portfolio, strategy=strat, instrument=goog, qty=Decimal("3.0"), avg_cost=Decimal("190.00"),
    )

    run, intents = _run_and_dispatch(portfolio)

    target = run.targets.get(instrument=goog)
    assert target.reason == "liquidate_disabled_allocation"
    assert target.delta_qty == Decimal("-3.0")
    assert len(intents) == 1
    assert LogicalPosition.objects.get(portfolio=portfolio, strategy=strat, instrument=goog).qty == Decimal("0.0")


# ---------- 3. Apalancamiento y estado de posición en vivo ----------

def _ohlcv(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    closes = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"Open": closes, "High": closes + 1, "Low": closes - 1, "Close": closes, "Volume": 1_000_000.0}, index=idx,
    )


# 3 trades ganadores cerrados y un 4º trade abierto al final
_PULSES = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, 0]
_CLOSES = [100, 100, 105, 110, 108, 108, 112, 118, 115, 115, 120, 126, 124, 124, 127, 129]


def _fake_strategy(df, params):
    sig = pd.Series(_PULSES[: len(df)], index=df.index, dtype=float)
    return sig, pd.DataFrame({"SignalType": "default"}, index=df.index)


def test_pulses_to_long_state_holds_between_entry_and_exit():
    state = pulses_to_long_state(pd.Series([0, 1, 0, 0, -1, 0, 1, 0]))
    assert state.tolist() == [False, True, True, True, False, False, True, True]


def test_live_leverage_matches_backtest_open_trade():
    df = _ohlcv(_CLOSES)
    state = pulses_to_long_state(pd.Series(_PULSES, index=df.index))

    res = BacktestEngine(initial_cash=2000.0)._simulate(
        df=df, signals=state.astype(int), symbol="QQQ",
        base_leverage=1.0, max_leverage=1.7, use_dynamic_leverage=True,
    )
    open_trade = res["trades"][-1]
    assert open_trade.get("open") is True

    live = current_trade_leverage(df["Close"].values, state.values, 1.0, 1.7, True)
    assert round(live, 2) == open_trade["leverage"]
    assert live == pytest.approx(1.7)  # sin pérdidas => PF 99 => apalancamiento máximo


def test_multi_signal_engine_applies_leverage_from_params():
    df = _ohlcv(_CLOSES)
    params = {
        "strategies": [{"strategy_name": "fake", "params": []}],
        "leverage": 1.0, "max_leverage": 1.7, "use_regimes": True,
    }
    with patch.dict("apps.strategies.engines.multi_signal.STRATEGY_MAP", {"fake": _fake_strategy}):
        targets, diag = MultiSignalEngine().compute_signal({"QQQ": df}, params, {"QQQ": Decimal("1.0")})
        assert targets["QQQ"] == Decimal("1.7")
        assert diag["QQQ"]["applied_leverage"] == 1.7

        # Sin apalancamiento dinámico se mantiene el apalancamiento base
        params["use_regimes"] = False
        targets, _ = MultiSignalEngine().compute_signal({"QQQ": df}, params, {"QQQ": Decimal("1.0")})
        assert targets["QQQ"] == Decimal("1.0")


def test_single_signal_engine_holds_position_between_entry_and_exit_pulses():
    """El último pulso es 0 (sin cambio) pero el trade sigue abierto: la exposición no debe caer a 0."""
    df = _ohlcv(_CLOSES)
    params = {"strategies": [{"strategy_name": "fake", "params": []}]}
    with patch.dict("apps.strategies.engines.single_signal.STRATEGY_MAP", {"fake": _fake_strategy}):
        targets, diag = SingleSignalEngine().compute_signal({"NAUT": df}, params, {"NAUT": Decimal("1.0")})
        assert diag["NAUT"]["raw_signal"] == 0
        assert targets["NAUT"] == Decimal("1.0")

        # Tras el pulso de salida queda plano
        flat_df = df.iloc[:12]
        targets, _ = SingleSignalEngine().compute_signal({"NAUT": flat_df}, params, {"NAUT": Decimal("1.0")})
        assert targets["NAUT"] == Decimal("0.0")


# ---------- 4. Envío de órdenes: mercado DAY, sin MOC ----------

@pytest.mark.django_db
def test_alpaca_dispatch_sends_fractional_day_market_order():
    from alpaca.trading.enums import TimeInForce
    from apps.brokers.models import BrokerAccount, BrokerCredential

    portfolio = _make_portfolio()
    qqq = Instrument.objects.create(symbol="QQQ", name="QQQ", asset_class="etf")
    _set_quote(qqq, "750.00")
    cred = BrokerCredential.objects.create(user=portfolio.user, provider="alpaca", environment="paper")
    cred.set_secrets({"key_id": "PKTEST", "secret_key": "SECRET"})
    cred.save()
    account = BrokerAccount.objects.create(
        portfolio=portfolio, credential=cred, provider="alpaca", environment="paper",
        execution_mode="auto", display_name="Alpaca Paper",
    )
    strat, ver = _make_strategy("qqq11", qqq)
    Allocation.objects.create(portfolio=portfolio, strategy=strat, target_weight=Decimal("0.5"), broker_account=account)
    _set_signal(ver, qqq, "1.35")

    run = RebalanceRun.objects.create(
        portfolio=portfolio, trigger="scheduled", as_of=timezone.now(), equity_base=Decimal("4000.00"),
    )
    intents = PortfolioAllocator(run).generate_targets_and_intents()

    at_1557 = datetime(2026, 10, 13, 15, 57, 5, tzinfo=NY_TZ)
    with patch("apps.trading.execution.get_ny_now", return_value=at_1557), \
            patch("alpaca.trading.client.TradingClient") as client_cls:
        client_cls.return_value.submit_order.return_value.id = "alpaca-order-1"
        order = OrderDispatcher().dispatch_intent(intents[0])

    req = client_cls.return_value.submit_order.call_args.kwargs["order_data"]
    assert req.time_in_force == TimeInForce.DAY
    assert req.qty == 3.6
    assert order.order_type == "market"
    assert order.status == "accepted"
    # La posición lógica solo se registra al reconciliar el fill real
    assert not LogicalPosition.objects.filter(portfolio=portfolio).exists()


# ---------- 5. Corrección de datos y reconciliación ----------

@pytest.mark.django_db
def test_reassign_logical_positions_command_merges_into_target_strategy():
    portfolio = _make_portfolio()
    qqq = Instrument.objects.create(symbol="QQQ", name="QQQ", asset_class="etf")
    goog = Instrument.objects.create(symbol="GOOG", name="GOOG", asset_class="equity")
    wrong_a, _ = _make_strategy("goog-zdema", goog, kind="one_strategy", engine="single_signal")
    wrong_b, _ = _make_strategy("naut-zdema", goog, kind="one_strategy", engine="single_signal")
    right, _ = _make_strategy("qqq11", qqq)
    LogicalPosition.objects.create(portfolio=portfolio, strategy=wrong_a, instrument=qqq, qty=Decimal("3.0"), avg_cost=Decimal("750.00"))
    LogicalPosition.objects.create(portfolio=portfolio, strategy=wrong_b, instrument=qqq, qty=Decimal("1.0"), avg_cost=Decimal("746.00"))

    args = ["reassign_logical_positions", "--portfolio", str(portfolio.id), "--symbol", "QQQ", "--to-strategy", "qqq11"]

    call_command(*args)  # dry-run: no cambia nada
    assert LogicalPosition.objects.filter(portfolio=portfolio).count() == 2
    assert not LogicalPosition.objects.filter(strategy=right).exists()

    call_command(*args, "--apply")
    pos = LogicalPosition.objects.get(portfolio=portfolio)
    assert pos.strategy == right
    assert pos.qty == Decimal("4.0")
    assert pos.avg_cost == Decimal("749.00")


def test_diff_positions_flags_only_real_mismatches():
    logical = {"QQQ": Decimal("2.68035434"), "NAUT": Decimal("10")}
    broker = {"QQQ": Decimal("5.87673177"), "NAUT": Decimal("10.00000001"), "IONQ": Decimal("1")}
    mismatches = diff_positions(logical, broker)
    assert [m[0] for m in mismatches] == ["IONQ", "QQQ"]
