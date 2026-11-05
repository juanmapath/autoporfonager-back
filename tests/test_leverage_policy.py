"""F4: politicas de apalancamiento (fixed / regime_pf) y proveedores de regimen."""
import numpy as np
import pandas as pd
import pytest

from apps.strategies.backtest import BacktestEngine
from apps.strategies.regimes.macro import MacroRegimeProvider
from apps.strategies.regimes.service import align_labels
from apps.strategies.signals.sizing import (
    ClosedTrade, LeveragePolicy, POLICY_FIXED, POLICY_REGIME_PF, current_trade_leverage,
    policy_entry_leverage, pulses_to_long_state, regime_pf_estimate, resolve_leverage_policy,
)


def make_series(seed, n=1500):
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


# ---------- Resolucion de la politica ----------

def test_policy_resolution():
    assert resolve_leverage_policy({}).type == POLICY_FIXED
    assert resolve_leverage_policy({"use_regimes": False}).type == POLICY_FIXED
    # use_regimes=true es regime_pf (ya no existe el PF global)
    assert resolve_leverage_policy({"use_regimes": True}).type == POLICY_REGIME_PF
    p = resolve_leverage_policy({"leverage_policy": {"type": "regime_pf"}})
    assert p.providers == ("macro",) and p.min_trades == 3 and p.lookback_years == 15 and p.combine == "min"
    for bad in ({"type": "global_pf"}, {"type": "nope"}, {"type": "regime_pf", "combine": "max"}):
        with pytest.raises(ValueError):
            resolve_leverage_policy({"leverage_policy": bad})


def test_fixed_policy_always_applies_base_leverage_in_backtest_and_live():
    df, sig = make_series(3)
    state = pulses_to_long_state(sig)
    result = BacktestEngine()._simulate(df, state.astype(int), "X", 1.0, 2.0)
    assert {t["leverage"] for t in result["trades"]} == {1.0}
    assert current_trade_leverage(df["Close"].values, state.values, 1.0, 2.0) == 1.0
    assert current_trade_leverage(df["Close"].values, state.values, 1.5, 2.0, policy=LeveragePolicy()) == 1.5


# ---------- regime_pf: reglas de la politica ----------

def _trades(spec, start="2020-01-01"):
    """spec: lista de (pnl, regimen). Un trade por semana, cerrado 3 dias despues de la entrada."""
    out = []
    for i, (pnl, label) in enumerate(spec):
        entry = pd.Timestamp(start) + pd.Timedelta(days=7 * i)
        out.append(ClosedTrade(entry, entry + pd.Timedelta(days=3), pnl, {"macro": label}))
    return out


POLICY = LeveragePolicy(type=POLICY_REGIME_PF, min_trades=8, lookback_years=15)
AS_OF = pd.Timestamp("2024-01-01")


def test_pf_of_current_regime_maps_to_leverage_tiers():
    strong = _trades([(10, "R_A")] * 9 + [(-2, "R_A")])        # PF 45 -> maximo
    medium = _trades([(3, "R_A")] * 4 + [(-2, "R_A")] * 4 + [(1, "R_A")] * 2)  # ganancia 14 / perdida 8 = 1.75 -> medio
    weak = _trades([(1, "R_A")] * 4 + [(-2, "R_A")] * 6)       # PF 0.33 -> base
    assert policy_entry_leverage(POLICY, strong, 1.0, 2.0, AS_OF, {"macro": "R_A"})[0] == 2.0
    assert policy_entry_leverage(POLICY, medium, 1.0, 2.0, AS_OF, {"macro": "R_A"})[0] == 1.5
    assert policy_entry_leverage(POLICY, weak, 1.0, 2.0, AS_OF, {"macro": "R_A"})[0] == 1.0


def test_trades_in_other_regimes_do_not_count_and_few_trades_use_base_leverage():
    # 12 trades ganadores en R_A, solo 3 en R_B: hoy estamos en R_B -> base (no se usa el PF de R_A)
    trades = _trades([(10, "R_A")] * 12 + [(10, "R_B")] * 3)
    lev, diag = policy_entry_leverage(POLICY, trades, 1.0, 2.0, AS_OF, {"macro": "R_B"})
    info = diag["providers"]["macro"]
    assert lev == 1.0 and info["source"] == "base_insufficient_trades" and info["n_trades_regime"] == 3
    lev_a, diag_a = policy_entry_leverage(POLICY, trades, 1.0, 2.0, AS_OF, {"macro": "R_A"})
    assert lev_a == 2.0 and diag_a["providers"]["macro"]["source"] == "regime_pf"


def test_no_current_regime_uses_base_leverage():
    trades = _trades([(10, "R_A")] * 10)
    lev, diag = policy_entry_leverage(POLICY, trades, 1.0, 2.0, AS_OF, {"macro": None})
    assert lev == 1.0 and diag["providers"]["macro"]["source"] == "base_no_regime"


def test_diagnostics_fields():
    trades = _trades([(3, "R_A")] * 6 + [(-1, "R_A")] * 2)
    info = regime_pf_estimate(POLICY, trades, "macro", "R_A", AS_OF)
    assert info == {"regime": "R_A", "n_trades_regime": 8, "pf_regime": pytest.approx(9.0), "pf_used": pytest.approx(9.0),
                    "source": "regime_pf"}


def test_lookback_excludes_old_trades():
    old = _trades([(100, "R_A")] * 10, start="2000-01-01")
    recent = _trades([(1, "R_A")] * 4 + [(-1, "R_A")] * 4, start="2022-01-01")
    pol = LeveragePolicy(type=POLICY_REGIME_PF, min_trades=8, lookback_years=15)
    info = regime_pf_estimate(pol, old + recent, "macro", "R_A", AS_OF)
    assert info["n_trades_regime"] == 8 and info["pf_regime"] == pytest.approx(1.0)
    all_info = regime_pf_estimate(LeveragePolicy(type=POLICY_REGIME_PF, min_trades=8, lookback_years=None),
                                  old + recent, "macro", "R_A", AS_OF)
    assert all_info["n_trades_regime"] == 18 and all_info["pf_regime"] > 50


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
    good = [10, 10, 10, 10, 10, 10, 10, 10, 10, -1]
    bad = [1, -5, -5, -5, -5, -5, -5, -5, -5, -5]
    trades = [
        ClosedTrade(pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * i), pd.Timestamp("2020-01-03") + pd.Timedelta(days=7 * i),
                    pnl, {"fake_a": "UP", "fake_b": "X"}) for i, pnl in enumerate(good)
    ] + [
        ClosedTrade(pd.Timestamp("2021-06-01") + pd.Timedelta(days=7 * i), pd.Timestamp("2021-06-03") + pd.Timedelta(days=7 * i),
                    pnl, {"fake_a": "DN", "fake_b": "Y"}) for i, pnl in enumerate(bad)
    ]
    common = dict(type=POLICY_REGIME_PF, min_trades=8, lookback_years=None)
    now = {"fake_a": "UP", "fake_b": "Y"}  # a sugiere el maximo (2.0); b sugiere el base (1.0)
    lev_min, diag = policy_entry_leverage(LeveragePolicy(providers=("fake_a", "fake_b"), combine="min", **common),
                                          trades, 1.0, 2.0, AS_OF, now)
    lev_mean, _ = policy_entry_leverage(LeveragePolicy(providers=("fake_a", "fake_b"), combine="mean", **common),
                                        trades, 1.0, 2.0, AS_OF, now)
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
    pol = LeveragePolicy(type=POLICY_REGIME_PF, min_trades=2, lookback_years=None)
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
    pol = LeveragePolicy(type=POLICY_REGIME_PF, min_trades=3, lookback_years=15)

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


# ---------- ranked_allocation usa la misma politica ----------

def _ranked_df(n=400, drift=0.003, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.004, n)))
    return pd.DataFrame({"Close": close}, index=idx)


def test_ranked_hold_mode_regime_pf_uses_daily_pf_in_current_regime_or_base():
    from apps.strategies.engines.ranked_allocation import RankedAllocationEngine as E

    df = _ranked_df()
    pol = LeveragePolicy(type=POLICY_REGIME_PF, lookback_years=None)
    series = {"macro": pd.Series(["R_1_0_0"] * len(df), index=df.index)}
    lev, info = E._compute_policy_leverage(df, None, pol, 1.0, 2.0, series)
    assert lev == 2.0 and info["mode"] == "hold_daily_pf" and info["regime"] == "R_1_0_0"
    assert info["n_trades_regime"] > 300 and info["pf_regime"] >= 1.35 and info["source"] == "regime_pf"

    # regimen vigente casi sin historia (< 60 dias) -> base
    short = {"macro": pd.Series(["R_0_0_0"] * (len(df) - 20) + ["R_1_1_1"] * 20, index=df.index)}
    lev, info = E._compute_policy_leverage(df, None, pol, 1.0, 2.0, short)
    assert lev == 1.0 and info["source"] == "base_insufficient_trades"

    # sin etiquetas (proveedor sin datos) -> base
    lev, info = E._compute_policy_leverage(df, None, pol, 1.0, 2.0, {})
    assert lev == 1.0 and info["source"] == "base_no_regime"


def test_ranked_strategy_mode_regime_pf_fixes_leverage_at_entry_of_open_trade():
    from apps.strategies.engines.ranked_allocation import RankedAllocationEngine as E

    df = _ranked_df(drift=0.004)
    n = len(df)
    sigs = pd.Series(0, index=df.index)
    for k in range(10, 300, 20):  # trades de 10 velas, todos ganadores con esta deriva
        sigs.iloc[k] = 1
        sigs.iloc[k + 10] = -1
    sigs.iloc[320] = 1  # trade abierto al final
    pol = LeveragePolicy(type=POLICY_REGIME_PF, min_trades=5, lookback_years=None)

    stable = {"macro": pd.Series(["R_A"] * n, index=df.index)}
    changed = {"macro": pd.Series(["R_A"] * 350 + ["R_B"] * (n - 350), index=df.index)}  # cambia DURANTE el trade abierto
    lev_a, info_a = E._compute_policy_leverage(df, sigs, pol, 1.0, 2.0, stable)
    lev_b, info_b = E._compute_policy_leverage(df, sigs, pol, 1.0, 2.0, changed)
    assert lev_a == lev_b == 2.0
    assert info_b["providers"]["macro"]["regime"] == "R_A"  # el regimen de la ENTRADA, no el actual
    assert info_a["mode"] == "strategy_trades" and info_a["providers"]["macro"]["n_trades_regime"] >= 5


@pytest.mark.django_db
def test_ranked_engine_reports_leverage_diagnostics_for_fixed_and_regime_pf():
    from apps.instruments.models import Instrument
    from apps.strategies.engines.ranked_allocation import RankedAllocationEngine

    Instrument.objects.create(symbol="AAA", name="AAA", asset_class="equity")
    df = _ranked_df()
    params = {"metrics": [], "rank_weights": [1.0], "execution": {"mode": "hold"}, "leverage": 1.0, "max_leverage": 2.0}
    _, diag = RankedAllocationEngine().compute_signal({"AAA": df}, dict(params), {})
    assert diag["AAA"]["leverage"]["leverage_policy"] == "fixed" and diag["AAA"]["applied_leverage"] == 1.0

    series = {"macro": pd.Series(["R_1_0_0"] * len(df), index=df.index)}
    pol = {"leverage_policy": {"type": "regime_pf", "lookback_years": None}}
    _, diag = RankedAllocationEngine().compute_signal({"AAA": df}, {**params, **pol, "_regime_series": series}, {})
    assert diag["AAA"]["leverage"]["leverage_policy"] == "regime_pf" and diag["AAA"]["applied_leverage"] == 2.0


# ---------- report, serializer, diagnosticos ----------

def test_regime_durations_counts_runs_and_calendar_days():
    from apps.strategies.regimes.report import regime_durations

    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    s = pd.Series(["A", "A", "B", "A", "A", "A"], index=idx)
    d = regime_durations(s)
    assert d["A"]["runs"] == 2 and d["A"]["avg_market_days"] == pytest.approx(2.5)
    assert d["B"]["runs"] == 1 and d["B"]["avg_market_days"] == 1 and d["B"]["avg_calendar_days"] == 1
    assert d["A"]["pct_of_time"] + d["B"]["pct_of_time"] == pytest.approx(100)


def test_leverage_policy_serializer_validation():
    from rest_framework import serializers
    from apps.strategies.serializers import validate_leverage_policy

    ok = validate_leverage_policy({"type": "regime_pf", "providers": ["macro"], "min_trades": 5})
    assert ok["type"] == "regime_pf"
    assert validate_leverage_policy(None) is None
    for bad in ({"type": "global_pf"}, {"type": "regime_pf", "providers": ["hmm"]}, {"type": "regime_pf", "combine": "max"},
                {"type": "layered", "caps": [{"key": "vol_target"}]}, {"type": "layered", "scalers": ["nope"]},
                {"type": "layered", "min": 2.0, "max": 1.5}):
        with pytest.raises(serializers.ValidationError):
            validate_leverage_policy(bad)


def test_single_signal_engine_reports_regime_pf_diagnostics(monkeypatch):
    from decimal import Decimal
    from apps.strategies.engines import single_signal
    from apps.strategies.engines.single_signal import SingleSignalEngine

    df, sig = make_series(3, n=900)
    sig = sig.copy()
    sig.iloc[-1] = 0
    state = pulses_to_long_state(sig)
    if not state.iloc[-1]:
        sig.iloc[-5] = 1  # asegura trade abierto al final
    monkeypatch.setitem(single_signal.STRATEGY_MAP, "Fake", lambda d, p: (sig, pd.DataFrame(index=d.index)))
    labels = pd.Series(["R_1_0_0"] * len(df), index=df.index)
    params = {
        "strategy_name": "Fake", "leverage": 1.0, "max_leverage": 2.0, "use_regimes": True,
        "leverage_policy": {"type": "regime_pf", "min_trades": 3, "lookback_years": None},
        "_regime_series": {"macro": labels},
    }
    targets, diag = SingleSignalEngine().compute_signal({"X": df}, params, {"X": Decimal("1")})
    d = diag["X"]
    assert d["is_long"]
    assert d["leverage"]["leverage_policy"] == "regime_pf"
    assert d["leverage"]["providers"]["macro"]["regime"] == "R_1_0_0"
    assert targets["X"] == Decimal(str(round(d["applied_leverage"], 4)))
