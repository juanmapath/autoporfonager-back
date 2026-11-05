"""04e: politica layered (escaladores, topes, etiquetas), simulador con costes de margen y estudio."""
import numpy as np
import pandas as pd
import pytest

from apps.strategies.leverage import layers as L
from apps.strategies.leverage.config import parse_layered
from apps.strategies.leverage.layers import LAYERS, LayerInputs, LayerSpec
from apps.strategies.leverage.simulator import SimCosts, live_layered_leverage, simulate
from apps.strategies.signals.sizing import LeveragePolicy, POLICY_FIXED, resolve_leverage_policy


def layered(**raw):
    return resolve_leverage_policy({"leverage": 1.0, "max_leverage": 2.0, "leverage_policy": {"type": "layered", **raw}})


def series(n=600, seed=0, drift=0.0006, vol=0.012, start="2012-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)


NO_COSTS = SimCosts(slippage=0.0, margin_interest=False)


@pytest.fixture
def fake_layers(monkeypatch):
    """Capas de prueba con valores constantes controlados."""
    def const(value):
        return lambda inputs, p, index, lmax: pd.Series(float(p.get("value", value)), index=index)

    monkeypatch.setitem(LAYERS, "fake_scaler", LayerSpec("fake_scaler", "scaler", "f", "", {"value": 1.6},
                                                         compute=const(1.6)))
    monkeypatch.setitem(LAYERS, "fake_cap", LayerSpec("fake_cap", "cap", "f", "", {"value": 1.3}, compute=const(1.3)))


# ---------- configuracion ----------

def test_layered_without_layers_equals_fixed_base():
    s = series()
    state = np.ones(len(s), dtype=bool)
    state[:10] = False
    res_l = simulate(s.values, state, s.index, layered(base=1.4, max=2.0, min=1.0), costs=NO_COSTS)
    res_f = simulate(s.values, state, s.index, LeveragePolicy(type=POLICY_FIXED), 1.4, 2.0, costs=NO_COSTS)
    np.testing.assert_allclose(res_l.equity, res_f.equity)
    assert res_l.resizes == 0 and set(np.round(res_l.applied[10:], 6)) == {1.4}


def test_validation_messages():
    for raw, msg in [
        ({"caps": [{"key": "vol_target"}]}, "tipo 'scaler'"),
        ({"scalers": [{"key": "nope"}]}, "inexistente"),
        ({"min": 2.0, "max": 1.5}, "no puede ser mayor"),
        ({"apply": "weekly"}, "apply invalido"),
    ]:
        with pytest.raises(ValueError, match=msg):
            parse_layered(raw)


def test_composition_scaler_times_cap_then_clip(fake_layers):
    s = series(100)
    state = np.ones(100, dtype=bool)
    # base 1 x 1.6 = 1.6, tope 1.3 -> 1.3
    res = simulate(s.values, state, s.index, layered(scalers=["fake_scaler"], caps=["fake_cap"]), costs=NO_COSTS)
    assert res.applied[0] == pytest.approx(1.3)
    assert res.last_info["active_layer"] == "fake_cap" and res.last_info["L_raw"] == pytest.approx(1.6)
    # sin tope: 1.6 (por debajo del maximo 2.0)
    res = simulate(s.values, state, s.index, layered(scalers=["fake_scaler"]), costs=NO_COSTS)
    assert res.applied[0] == pytest.approx(1.6)
    # escalador 3.0 -> clip al maximo
    res = simulate(s.values, state, s.index, layered(scalers=[{"key": "fake_scaler", "params": {"value": 3.0}}]),
                   costs=NO_COSTS)
    assert res.applied[0] == pytest.approx(2.0) and res.last_info["active_layer"] == "L_max"
    # tope 0.4 con min 0.5 -> 0.5 (por debajo del mercado)
    res = simulate(s.values, state, s.index, layered(min=0.5, caps=[{"key": "fake_cap", "params": {"value": 0.4}}]),
                   costs=NO_COSTS)
    assert res.applied[0] == pytest.approx(0.5) and res.last_info["active_layer"] == "L_min"


# ---------- causalidad de cada filtro ----------

def _inputs(n=700):
    out = {"self": series(n, 1)}
    for i, sym in enumerate(["^VIX", "^VIX3M", "HYG", "LQD", "SPY", "TLT", "GLD", "UUP", "QQQ"]):
        vol = 0.06 if sym.startswith("^") else 0.01
        out[sym] = series(n, 10 + i, drift=0.0, vol=vol) / (5 if sym.startswith("^") else 1)
    return LayerInputs(closes=out)


@pytest.mark.parametrize("key", ["vol_target", "trend_gate", "vix_term", "vix_relative", "credit_stress", "turbulence"])
def test_each_static_layer_is_causal(key):
    inputs = _inputs()
    index = inputs.closes["self"].index
    spec = LAYERS[key]
    full = spec.compute(inputs, dict(spec.defaults), index, 2.0)
    cut = 500
    trunc_inputs = LayerInputs(closes={k: v.iloc[:cut] for k, v in inputs.closes.items()})
    trunc = spec.compute(trunc_inputs, dict(spec.defaults), index[:cut], 2.0)
    pd.testing.assert_series_equal(trunc, full.iloc[:cut], check_names=False)


def test_trend_gate_hysteresis_band():
    idx = pd.bdate_range("2020-01-01", periods=8)
    # SMA(2) con banda 5%: el estado solo cambia al salir de la banda
    close = pd.Series([100, 100, 103, 106, 103, 99, 96, 99], index=idx, dtype=float)
    cap = L._trend_gate(LayerInputs({"self": close}), {"sma": 2, "band": 0.05, "cap_below": 1.0}, idx, 2.0)
    # sma: -,100,101.5,104.5,104.5,101,97.5,97.5 ; 99 < 101*0.95? no (95.95) -> sigue encendido;
    # 96 < 97.5*0.95 = 92.6? no -> encendido. Con bajadas mayores se apaga.
    assert cap.tolist() == [2.0] * 8
    close2 = pd.Series([100, 100, 90, 80, 85, 90, 100, 110], index=idx, dtype=float)
    cap2 = L._trend_gate(LayerInputs({"self": close2}), {"sma": 2, "band": 0.05, "cap_below": 1.0}, idx, 2.0)
    # 90 < 100*0.95 -> apagado; 85 (sma 82.5) no supera 82.5*1.05=86.6 -> sigue apagado; 90 > 87.5*1.05=91.9? no;
    # 100 > 95*1.05=99.75 -> encendido
    assert cap2.tolist() == [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]


# ---------- apply daily, min_change, max_step_up ----------

def test_daily_resize_respects_min_change_and_asymmetric_steps(monkeypatch):
    n = 40
    idx = pd.bdate_range("2021-01-01", periods=n)
    values = [1.0] * 5 + [1.1] * 5 + [2.0] * 20 + [1.2] * 10
    monkeypatch.setitem(LAYERS, "step_scaler", LayerSpec(
        "step_scaler", "scaler", "s", "", {}, compute=lambda i, p, index, lmax: pd.Series(values, index=index)))
    pol = layered(scalers=["step_scaler"], min_change=0.15, max_step_up=0.25, step_days=5)
    res = simulate(np.full(n, 100.0), np.ones(n, dtype=bool), idx, pol, costs=NO_COSTS)
    applied = res.applied
    assert list(applied[:10]) == [1.0] * 10            # 1.1: cambio < 0.15, no redimensiona
    # sube como mucho 0.25 cada 5 dias: 1.25 (dia 10), 1.5 (15), 1.75 (20), 2.0 (25)
    assert applied[10] == pytest.approx(1.25) and applied[14] == pytest.approx(1.25)
    assert applied[15] == pytest.approx(1.5) and applied[20] == pytest.approx(1.75) and applied[25] == pytest.approx(2.0)
    assert applied[30] == pytest.approx(1.2)          # la bajada es inmediata
    assert res.resizes == 5


def test_entry_apply_keeps_leverage_during_the_trade(monkeypatch):
    n = 30
    idx = pd.bdate_range("2021-01-01", periods=n)
    values = [1.0] * 10 + [2.0] * 20
    monkeypatch.setitem(LAYERS, "step_scaler", LayerSpec(
        "step_scaler", "scaler", "s", "", {}, compute=lambda i, p, index, lmax: pd.Series(values, index=index)))
    res = simulate(np.full(n, 100.0), np.ones(n, dtype=bool), idx, layered(scalers=["step_scaler"], apply="entry"),
                   costs=NO_COSTS)
    assert set(res.applied) == {1.0} and res.resizes == 0


def test_drawdown_brake_reduces_leverage_with_the_bot_drawdown():
    idx = pd.bdate_range("2021-01-01", periods=6)
    closes = np.array([100, 100, 90, 80, 80, 80], dtype=float)
    pol = layered(base=2.0, max=2.0, caps=[{"key": "drawdown_brake", "params": {"dd_limit": 0.5}}], min_change=0.01,
                  max_step_up=5)
    res = simulate(closes, np.ones(6, dtype=bool), idx, pol, costs=NO_COSTS)
    # dia 2: precio -10% con 2x -> equity -20% -> tope 2 x (1 - 0.2/0.5) = 1.2
    assert res.applied[1] == pytest.approx(2.0)
    assert res.applied[2] == pytest.approx(1.2, abs=1e-6)
    assert res.last_info["caps"]["drawdown_brake"] < 2.0


# ---------- costes: interes y llamada de margen ----------

def test_margin_interest_charged_on_borrowed_cash():
    n = 253
    idx = pd.bdate_range("2021-01-01", periods=n)
    irx = pd.Series(3.0, index=idx)  # 3% anual
    pol = LeveragePolicy(type=POLICY_FIXED)
    res = simulate(np.full(n, 100.0), np.ones(n, dtype=bool), idx, pol, 2.0, 2.0,
                   inputs=LayerInputs({"^IRX": irx}), costs=SimCosts(slippage=0.0, margin_spread=0.02))
    # presta 1.0 (equity 1, posicion 2) a 5% anual durante 252 dias ~ 0.05 (compuesto)
    assert res.interest_paid == pytest.approx(np.exp(0.05) - 1, rel=0.01)
    assert res.equity[-1] == pytest.approx(1 - res.interest_paid, rel=1e-6)


def test_margin_call_liquidates_and_waits_for_a_new_entry():
    idx = pd.bdate_range("2021-01-01", periods=8)
    closes = np.array([100, 100, 80, 70, 72, 75, 80, 85], dtype=float)
    state = np.array([1, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    # 2x: equity/posicion < 0.30 cuando el precio cae ~28.6%: el dia 3 (70)
    res = simulate(closes, state, idx, LeveragePolicy(type=POLICY_FIXED), 2.0, 2.0, costs=NO_COSTS)
    assert res.margin_calls == [str(idx[3].date())]
    assert res.trades[0]["forced"] is True
    assert res.applied[4] == 0.0              # no reentra con la misma senal
    assert res.applied[6] == pytest.approx(2.0)  # nueva entrada tras la salida


# ---------- paridad vivo / backtest ----------

def test_live_and_backtest_layered_give_the_same_leverage():
    inputs = _inputs(700)
    s = inputs.closes["self"]
    state = np.zeros(len(s), dtype=bool)
    for a in range(30, 680, 45):
        state[a:a + 35] = True
    pol = layered(scalers=[{"key": "vol_target", "params": {"target_vol": 0.30}}],
                  caps=[{"key": "trend_gate", "params": {"sma": 50}}, "vix_relative", "drawdown_brake"])
    full = simulate(s.values, state, s.index, pol, inputs=inputs)
    shared = LayerInputs({k: v for k, v in inputs.closes.items() if k != "self"})
    assert full.resizes > 0
    for k in (100, 250, 400, 555, 699):
        if not state[k]:
            continue
        lev, info = live_layered_leverage(s.values[:k + 1], state[:k + 1], s.index[:k + 1], pol,
                                          LayerInputs({kk: vv.iloc[:k + 1] for kk, vv in shared.closes.items()}))
        assert lev == pytest.approx(full.applied[k]), k
        assert info["leverage_policy"] == "layered" and "active_layer" in info


def test_rebalance_policy_resizes_on_leverage_change():
    from decimal import Decimal
    from types import SimpleNamespace
    from apps.trading.rebalance_policy import decide_position_trade

    p = SimpleNamespace(rebalance_tolerance=Decimal("0.05"), rebalance_tolerance_mode="relative")
    args = (p, Decimal("10"), Decimal("12"), Decimal("100"), Decimal("10000"))
    assert decide_position_trade(*args, is_rebalance_date=False) == (False, "hold_until_rebalance_date")
    assert decide_position_trade(*args, is_rebalance_date=False, leverage_change=True) == (True, "leverage_change")


# ---------- politicas de etiqueta ----------

def _label_setup(rets_by_regime, n_pad=5):
    """Construye precios/estado con trades de 1 dia y retornos dados; todos en el regimen indicado."""
    closes, state, labels = [100.0], [False], ["R_A"]
    for ret, lab in rets_by_regime:
        closes += [closes[-1], closes[-1] * (1 + ret)]
        state += [True, False]
        labels += [lab, lab]
    closes += [closes[-1]] * n_pad
    state += [False] * (n_pad - 1) + [True]
    labels += ["R_A"] * n_pad
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    return np.array(closes), np.array(state), idx, {"macro": labels}


def test_macro_pf_label_uses_regime_pf_of_unlevered_trades():
    closes, state, idx, labels = _label_setup([(0.02, "R_A")] * 4 + [(-0.01, "R_A")])
    pol = layered(label_policies=[{"key": "macro_pf", "params": {"min_trades": 3}}])
    res = simulate(closes, state, idx, pol, regime_labels=labels, costs=NO_COSTS)
    info = res.last_info["labels"]["macro_pf"]
    assert info["regime"] == "R_A" and info["n_trades_regime"] == 5 and info["pf_regime"] == pytest.approx(8.0, rel=1e-3)
    assert res.applied[-1] == pytest.approx(2.0)


def test_kelly_regime_label_and_fallback():
    closes, state, idx, labels = _label_setup([(0.02, "R_A"), (0.01, "R_A"), (-0.005, "R_A"), (0.015, "R_A")])
    pol = layered(label_policies=[{"key": "kelly_regime", "params": {"fraction": 0.5, "min_trades": 3}}])
    res = simulate(closes, state, idx, pol, regime_labels=labels, costs=NO_COSTS)
    rets = [0.02, 0.01, -0.005, 0.015]
    expected = min(2.0, 0.5 * np.mean(rets) / np.var(rets, ddof=1))
    assert res.last_info["labels"]["kelly_regime"]["kelly"] == pytest.approx(0.5 * np.mean(rets) / np.var(rets, ddof=1), rel=1e-3)
    assert res.applied[-1] == pytest.approx(expected)
    # regimen sin trades -> base
    labels_b = {"macro": labels["macro"][:-1] + ["R_B"]}
    res_b = simulate(closes, state, idx, pol, regime_labels=labels_b, costs=NO_COSTS)
    assert res_b.applied[-1] == pytest.approx(1.0)
    assert res_b.last_info["labels"]["kelly_regime"]["source"] == "base_insufficient_trades"


def test_single_signal_engine_with_layered_policy(monkeypatch):
    from decimal import Decimal
    from apps.strategies.engines import single_signal
    from apps.strategies.engines.single_signal import SingleSignalEngine

    s = series(400, 3)
    df = pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1.0})
    sig = pd.Series(0, index=df.index)
    sig.iloc[100] = 1
    monkeypatch.setitem(single_signal.STRATEGY_MAP, "Fake", lambda d, p: (sig.reindex(d.index).fillna(0),
                                                                       pd.DataFrame(index=d.index)))
    params = {"strategy_name": "Fake", "leverage": 1.0, "max_leverage": 2.0,
              "leverage_policy": {"type": "layered", "scalers": [{"key": "vol_target", "params": {"target_vol": 0.3}}]},
              "_layer_inputs": LayerInputs({})}
    targets, diag = SingleSignalEngine().compute_signal({"X": df}, params, {"X": Decimal("1")})
    d = diag["X"]
    assert d["is_long"] and d["leverage"]["leverage_policy"] == "layered"
    assert 1.0 <= d["applied_leverage"] <= 2.0 and "vol_target" in d["leverage"]["scalers"]
    assert targets["X"] == Decimal(str(round(d["applied_leverage"], 4)))
    assert isinstance(d["leverage_resized"], bool)


# ---------- estudio ----------

def test_leverage_study_on_synthetic_data_applies_the_acceptance_criterion():
    from apps.strategies.leverage.report_html import render_report
    from apps.strategies.leverage.study import LeverageStudy, StudyConfig

    rng = np.random.default_rng(7)
    n = 2600
    idx = pd.bdate_range("2007-01-01", periods=n)
    rets = rng.normal(0.0005, 0.011, n)
    rets[300:340] = -0.02   # crisis en la primera mitad
    rets[1900:1930] = -0.025  # crisis en la segunda mitad
    closes = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    state = np.ones(n, dtype=bool)  # siempre dentro: 1x replica exactamente al benchmark (sin slippage)
    study = LeverageStudy.from_data(closes, state, 1.0, 2.0, start="2007-01-01", split="2012-01-01",
                                    costs=SimCosts(slippage=0.0, margin_interest=False))
    configs = [
        StudyConfig("1x fijo", "referencia", {"type": "fixed"}, 1.0, 1.0),
        StudyConfig("2x fijo", "referencia", {"type": "fixed"}, 2.0, 2.0),
        StudyConfig("vol_target 0.10", "vol_target",
                    {"type": "layered", "min": 0.5, "max": 2.0,
                     "scalers": [{"key": "vol_target", "params": {"target_vol": 0.10}}]}, n_layers=1),
    ]
    out = study.run_grid(configs)
    by = {r.config.name: r for r in out["results"]}
    assert by["1x fijo"].acceptable_both                       # igual al benchmark
    assert not by["2x fijo"].acceptable_both                   # cae el doble que el benchmark
    assert by["2x fijo"].full["margin_calls"] == 1            # 2x con caidas del 55% -> llamada de margen
    assert any("llamada" in x for x in by["2x fijo"].full["reasons"])
    assert by["2x fijo"].full["crises"]                          # ventanas de crisis detectadas
    assert out["selection"]["winner"] is not None and out["selection"]["winner"].acceptable_both
    assert out["n_configs"] == 3 and out["n_runs"] >= 3
    page = render_report(study, out, "Estudio sintetico")
    assert "<svg" in page and "Recomendada" in page and "2x fijo" in page


def test_leverage_catalog_lists_policies_layers_and_providers():
    from apps.strategies.leverage.catalog import leverage_catalog

    cat = leverage_catalog()
    assert [p["key"] for p in cat["policies"]] == ["fixed", "global_pf", "regime_pf", "layered"]
    kinds = {l["key"]: l["kind"] for l in cat["layers"]}
    assert kinds["vol_target"] == "scaler" and kinds["trend_gate"] == "cap" and kinds["macro_pf"] == "label"
    assert {p["key"]: p["available"] for p in cat["providers"]}["macro"] is True
