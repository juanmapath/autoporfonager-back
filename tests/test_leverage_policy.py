"""F4: politicas de apalancamiento (fixed / global_pf / regime_pf) y proveedores de regimen."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apps.strategies.backtest import BacktestEngine
from apps.strategies.regimes.macro import MacroRegimeProvider
from apps.strategies.regimes.service import align_labels
from apps.strategies.signals.sizing import (
    ClosedTrade, LeveragePolicy, POLICY_GLOBAL_PF, POLICY_REGIME_PF, current_trade_leverage,
    policy_entry_leverage, pulses_to_long_state, regime_pf_estimate, resolve_leverage_policy,
)

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "leverage_global_pf_golden.json").read_text())


def make_series(seed, n=1500):
    """Misma construccion con la que se capturo el golden (codigo ANTERIOR a F4)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0009, 0.012, n)))
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}, index=idx)
    sig = np.zeros(n, dtype=int)
    i, inpos = 0, False
    while i < n:
        i += int(rng.integers(5, 40))
        if i >= n:
            break
        sig[i] = -1 if inpos else 1
        inpos = not inpos
    return df, pd.Series(sig, index=idx)


# ---------- Regresion numerica: global_pf == apalancamiento actual (live y backtest) ----------

@pytest.mark.parametrize("seed", [3, 4, 5])
@pytest.mark.parametrize("variant", ["legacy_flag", "policy_global_pf", "resolved_from_use_regimes"])
def test_global_pf_reproduces_current_leverage_in_backtest_and_live(seed, variant):
    gold = GOLDEN[str(seed)]
    df, sig = make_series(seed)
    state = pulses_to_long_state(sig)

    if variant == "legacy_flag":
        kwargs, live_kwargs = dict(use_dynamic_leverage=True), dict(use_dynamic_leverage=True)
    else:
        params = {"leverage_policy": {"type": "global_pf"}} if variant == "policy_global_pf" else {"use_regimes": True}
        policy = resolve_leverage_policy(params)
        assert policy.type == POLICY_GLOBAL_PF
        kwargs = dict(use_dynamic_leverage=False, leverage_policy=policy)
        live_kwargs = dict(policy=policy, dates=df.index)

    result = BacktestEngine()._simulate(df, sig, "X", 1.0, 2.0, **kwargs)
    assert [t["leverage"] for t in result["trades"]] == gold["trade_leverages"]
    assert result["final_value"] == gold["final"]

    live = [
        current_trade_leverage(df["Close"].values[:k], state.values[:k], 1.0, 2.0, **live_kwargs)
        for k in range(60, len(df), 97)
    ]
    assert live == gold["live"]


def test_policy_resolution_is_backward_compatible():
    assert resolve_leverage_policy({}).type == "fixed"
    assert resolve_leverage_policy({"use_regimes": False}).type == "fixed"
    assert resolve_leverage_policy({"use_regimes": True}).type == "global_pf"
    p = resolve_leverage_policy({"use_regimes": True, "leverage_policy": {"type": "regime_pf"}})
    assert p.type == "regime_pf" and p.providers == ("macro",) and p.min_trades == 8
    assert p.shrinkage_k == 10 and p.lookback_years == 15 and p.combine == "min"
    with pytest.raises(ValueError):
        resolve_leverage_policy({"leverage_policy": {"type": "nope"}})


# ---------- regime_pf: reglas de la politica ----------

def _trades(spec, start="2020-01-01"):
    """spec: lista de (pnl, regimen). Un trade por semana, cerrado 3 dias despues de la entrada."""
    out = []
    for i, (pnl, label) in enumerate(spec):
        entry = pd.Timestamp(start) + pd.Timedelta(days=7 * i)
        out.append(ClosedTrade(entry, entry + pd.Timedelta(days=3), pnl, {"macro": label}))
    return out


POLICY = LeveragePolicy(type=POLICY_REGIME_PF, min_trades=8, shrinkage_k=10, lookback_years=15)
AS_OF = pd.Timestamp("2024-01-01")


def test_regime_with_fewer_than_min_trades_falls_back_to_global_pf():
    # 12 trades en R_A (PF global alto), 3 en R_B (PF 0.2): hoy estamos en R_B -> PF global
    trades = _trades([(10, "R_A")] * 10 + [(-5, "R_A")] * 2 + [(1, "R_B"), (-5, "R_B"), (-5, "R_B")])
    info = regime_pf_estimate(POLICY, trades, "macro", "R_B", AS_OF)
    assert info["source"] == "pf_global" and info["n_trades_regime"] == 3
    assert info["pf_used"] == pytest.approx(info["pf_global"])
    lev, _ = policy_entry_leverage(POLICY, trades, 1.0, 2.0, AS_OF, {"macro": "R_B"})
    assert lev == 2.0  # PF global = 100 / 20 = 5 -> maximo, no el PF malo de R_B


def test_not_enough_total_trades_uses_base_leverage():
    trades = _trades([(10, "R_A")] * 5)
    lev, diag = policy_entry_leverage(POLICY, trades, 1.0, 2.0, AS_OF, {"macro": "R_A"})
    assert lev == 1.0 and diag["providers"]["macro"]["source"] == "base_insufficient_history"


def test_unknown_current_regime_uses_global_pf():
    trades = _trades([(10, "R_A")] * 9 + [(-2, "R_A")])
    info = regime_pf_estimate(POLICY, trades, "macro", None, AS_OF)
    assert info["source"] == "pf_global"


def test_shrinkage_moves_between_regime_pf_and_global_pf():
    def build(n_reg):
        # Global: PF 2.0 con muchos trades en R_A; el regimen R_B tiene PF 5.0 con n_reg trades
        base = [(2, "R_A")] * 20 + [(-1, "R_A")] * 20
        reg = [(5, "R_B")] * n_reg + [(-1, "R_B")] * (n_reg // 5) if n_reg else []
        return _trades(base + reg)

    small = regime_pf_estimate(POLICY, build(10), "macro", "R_B", AS_OF)     # n=12
    large = regime_pf_estimate(POLICY, build(1000), "macro", "R_B", AS_OF)  # n=1200
    assert small["pf_shrunk"] < large["pf_shrunk"]
    assert abs(large["pf_shrunk"] - min(large["pf_regime"], 10)) < 0.1     # n grande -> PF del regimen
    assert abs(small["pf_shrunk"] - small["pf_global"]) < abs(small["pf_shrunk"] - small["pf_regime"])  # n chico -> global


def test_shrinkage_formula_exact():
    trades = _trades([(3, "R_A")] * 8 + [(-1, "R_A")] * 4 + [(2, "R_B")] * 6 + [(-1, "R_B")] * 3)
    info = regime_pf_estimate(LeveragePolicy(type=POLICY_REGIME_PF, min_trades=8, shrinkage_k=10, lookback_years=None),
                              trades, "macro", "R_A", AS_OF)
    n, pf_reg, pf_glob = info["n_trades_regime"], info["pf_regime"], info["pf_global"]
    assert n == 12
    assert info["pf_shrunk"] == pytest.approx((n * pf_reg + 10 * pf_glob) / (n + 10))


def test_lookback_excludes_old_trades():
    old = _trades([(100, "R_A")] * 10, start="2000-01-01")
    recent = _trades([(1, "R_A")] * 8 + [(-1, "R_A")] * 8, start="2022-01-01")
    pol = LeveragePolicy(type=POLICY_REGIME_PF, min_trades=8, lookback_years=15)
    info = regime_pf_estimate(pol, old + recent, "macro", "R_A", AS_OF)
    assert info["n_trades_global"] == 16 and info["pf_global"] == pytest.approx(1.0)
    all_info = regime_pf_estimate(LeveragePolicy(type=POLICY_REGIME_PF, min_trades=8, lookback_years=None),
                                  old + recent, "macro", "R_A", AS_OF)
    assert all_info["n_trades_global"] == 26 and all_info["pf_global"] > 50


def test_trade_closing_after_decision_date_does_not_change_leverage():
    trades = _trades([(10, "R_A")] * 9 + [(-1, "R_A")])
    as_of = trades[-1].exit_date  # el ultimo trade cierra exactamente hoy: no cuenta
    before, _ = policy_entry_leverage(POLICY, trades[:-1], 1.0, 2.0, as_of, {"macro": "R_A"})
    with_future, _ = policy_entry_leverage(POLICY, trades, 1.0, 2.0, as_of, {"macro": "R_A"})
    assert before == with_future

    # y un trade enorme que cierra despues tampoco
    future = ClosedTrade(as_of - pd.Timedelta(days=5), as_of + pd.Timedelta(days=30), -10_000, {"macro": "R_A"})
    lev_future, _ = policy_entry_leverage(POLICY, trades[:-1] + [future], 1.0, 2.0, as_of, {"macro": "R_A"})
    assert lev_future == before


def test_combine_min_and_mean_with_two_fake_providers():
    good = [ClosedTrade(pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * i),
                        pd.Timestamp("2020-01-03") + pd.Timedelta(days=7 * i), pnl,
                        {"fake_a": "UP", "fake_b": "DOWN"})
            for i, pnl in enumerate([10] * 9 + [-1])]
    # fake_a: regimen UP tiene los 10 trades (PF alto) -> max; fake_b: DOWN idem pero hoy estamos en "NEW" (0 trades) -> PF global
    common = dict(type=POLICY_REGIME_PF, min_trades=8, shrinkage_k=0, lookback_years=None)
    only_a, _ = policy_entry_leverage(LeveragePolicy(providers=("fake_a",), **common), good, 1.0, 2.0, AS_OF,
                                      {"fake_a": "UP"})
    assert only_a == 2.0
    bad = [ClosedTrade(t.entry_date, t.exit_date, t.pnl, {"fake_a": "UP", "fake_b": "BAD"}) for t in good]
    bad += [ClosedTrade(pd.Timestamp("2021-06-01") + pd.Timedelta(days=7 * i),
                        pd.Timestamp("2021-06-03") + pd.Timedelta(days=7 * i), pnl,
                        {"fake_a": "UP", "fake_b": "BAD"}) for i, pnl in enumerate([-5] * 9 + [1])]
    # Con shrinkage_k=0, PF* = PF del regimen. fake_a (UP, todos los trades) y fake_b (BAD, todos) ven lo mismo:
    # fabricamos la diferencia con regimenes distintos
    mixed = [ClosedTrade(t.entry_date, t.exit_date, t.pnl, {"fake_a": "UP", "fake_b": "X"}) for t in good] + \
            [ClosedTrade(t.entry_date, t.exit_date, t.pnl, {"fake_a": "DN", "fake_b": "Y"}) for t in bad[10:]]
    pol_min = LeveragePolicy(providers=("fake_a", "fake_b"), combine="min", **common)
    pol_mean = LeveragePolicy(providers=("fake_a", "fake_b"), combine="mean", **common)
    now = {"fake_a": "UP", "fake_b": "Y"}  # a sugiere max (2.0), b sugiere base (1.0: PF de Y es ~0.2)
    lev_min, diag = policy_entry_leverage(pol_min, mixed, 1.0, 2.0, AS_OF, now)
    lev_mean, _ = policy_entry_leverage(pol_mean, mixed, 1.0, 2.0, AS_OF, now)
    assert diag["providers"]["fake_a"]["regime"] == "UP" and diag["providers"]["fake_b"]["regime"] == "Y"
    assert lev_min == 1.0
    assert lev_mean == pytest.approx(1.5)


def test_leverage_is_fixed_at_entry_even_if_regime_changes_during_the_trade():
    n = 400
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    state = np.zeros(n, dtype=bool)
    for a, b in [(10, 20), (30, 40), (50, 60), (70, 80), (90, 100), (110, 120), (130, 140), (150, 160),
                 (170, 180), (190, 200), (210, 220), (230, 240), (250, 399)]:
        state[a:b] = True
    pol = LeveragePolicy(type=POLICY_REGIME_PF, min_trades=2, shrinkage_k=1, lookback_years=None)
    labels_a = {"macro": ["R_A"] * n}
    labels_b = {"macro": ["R_A"] * 300 + ["R_B"] * (n - 300)}  # el regimen cambia en medio del ultimo trade (abierto)
    lev_a = current_trade_leverage(closes, state, 1.0, 2.0, policy=pol, dates=idx, regime_labels=labels_a)
    lev_b = current_trade_leverage(closes, state, 1.0, 2.0, policy=pol, dates=idx, regime_labels=labels_b)
    assert lev_a == lev_b


def test_live_and_backtest_give_the_same_entry_leverage_for_same_data():
    df, sig = make_series(3)
    state = pulses_to_long_state(sig)
    idx = df.index
    rng = np.random.default_rng(11)
    labels = list(rng.choice(["R_0_0_0", "R_1_0_0", "R_1_1_0"], size=len(idx)))
    pol = LeveragePolicy(type=POLICY_REGIME_PF, min_trades=3, shrinkage_k=4, lookback_years=15)

    # _simulate recibe el estado continuo (1 dentro / 0 fuera), como hace run_for_config
    result = BacktestEngine()._simulate(
        df, state.astype(int), "X", 1.0, 2.0, leverage_policy=pol, regime_labels={"macro": labels}
    )
    assert len({t["leverage"] for t in result["trades"]}) > 1  # la politica realmente varia el apalancamiento

    entry_pos = {str(d.date()): i for i, d in enumerate(idx)}
    for trade in result["trades"]:
        e = entry_pos[trade["entry_date"]]
        live = current_trade_leverage(
            df["Close"].values[:e + 1], state.values[:e + 1], 1.0, 2.0,
            policy=pol, dates=idx[:e + 1], regime_labels={"macro": labels[:e + 1]},
        )
        assert round(live, 2) == trade["leverage"], trade["entry_date"]


# ---------- MacroRegimeProvider ----------

def _macro_bars(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)

    def frame(drift, vol, start):
        c = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0}, index=idx)

    return {"HYG": frame(0.0002, 0.004, 80), "LQD": frame(0.0001, 0.004, 120),
            "DX-Y.NYB": frame(0.0, 0.005, 95), "^VIX": frame(0.0, 0.05, 18)}


def test_macro_regime_provider_is_causal():
    bars = _macro_bars()
    prov = MacroRegimeProvider()
    full = prov.compute(bars, {})
    cut = 300
    truncated = prov.compute({k: v.iloc[:cut] for k, v in bars.items()}, {})
    assert truncated.index.equals(full.index[:cut])
    pd.testing.assert_series_equal(truncated, full.iloc[:cut], check_names=False)
    # el primer tramo sin datos suficientes no se rellena hacia atras
    assert full.iloc[:40].isna().all()
    assert set(full.dropna().unique()) <= {f"R_{a}_{b}_{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    assert prov.required_symbols() == ["HYG", "LQD", "DX-Y.NYB", "^VIX"]


def test_align_labels_is_causal_and_uses_last_known_value():
    s = pd.Series(["A", "B"], index=pd.to_datetime(["2024-01-02", "2024-01-05"]))
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])
    assert align_labels(s, idx) == [None, "A", "A", "B", "B"]


@pytest.mark.django_db
def test_save_snapshots_persists_the_day_label_once():
    from apps.strategies.models import RegimeSnapshot
    from apps.strategies.regimes.service import save_snapshots

    bars = _macro_bars()
    out = save_snapshots({"macro": {}}, bars)
    assert set(out) == {"macro"}
    save_snapshots({"macro": {}}, bars)  # idempotente
    snap = RegimeSnapshot.objects.get(provider="macro")
    assert snap.label == out["macro"] and "Z_Risk" in snap.features
    assert RegimeSnapshot.objects.count() == 1
