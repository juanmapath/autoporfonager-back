"""
Lógica compartida entre el backtest y los motores en vivo para que ambos
dimensionen las posiciones exactamente igual:
  - conversión de pulsos discretos (1 / -1 / 0) a estado continuo de posición
  - apalancamiento dinámico walk-forward según Profit Factor acumulado
"""
from typing import Any, Dict, Sequence, Tuple
import numpy as np
import pandas as pd

DEFAULT_SLIPPAGE_PCT = 0.0005

PF_MAX_LEVERAGE_THRESHOLD = 2.5
PF_MID_LEVERAGE_THRESHOLD = 1.5
MIN_TRADES_FOR_DYNAMIC_LEVERAGE = 2


def pulses_to_long_state(signals: pd.Series) -> pd.Series:
    """Convierte pulsos (1: entrada, -1: salida, 0: sin cambio) a estado continuo (True: dentro)."""
    state = np.zeros(len(signals), dtype=bool)
    current = False
    for i, v in enumerate(signals.values):
        if v == 1:
            current = True
        elif v == -1:
            current = False
        state[i] = current
    return pd.Series(state, index=signals.index)


def resolve_leverage_params(params: Dict[str, Any]) -> Tuple[float, float, bool]:
    """Extrae (base_leverage, max_leverage, use_dynamic_leverage) de los params de una StrategyVersion."""
    params = params or {}
    base = float(params.get("leverage") or 1.0)
    max_lev = float(params.get("max_leverage") or base)
    return base, max(base, max_lev), bool(params.get("use_regimes", False))


def dynamic_leverage_from_pnls(
    closed_pnls: Sequence[float],
    base_leverage: float,
    max_leverage: float,
    use_dynamic_leverage: bool,
) -> float:
    """Apalancamiento a aplicar en una nueva entrada según el Profit Factor de los trades ya cerrados."""
    active_leverage = base_leverage
    if use_dynamic_leverage and len(closed_pnls) >= MIN_TRADES_FOR_DYNAMIC_LEVERAGE:
        sum_gains = sum(p for p in closed_pnls if p > 0)
        sum_losses = sum(abs(p) for p in closed_pnls if p <= 0)
        running_pf = sum_gains / sum_losses if sum_losses > 0 else (99.0 if sum_gains > 0 else 1.0)

        if running_pf >= PF_MAX_LEVERAGE_THRESHOLD:
            active_leverage = max_leverage
        elif running_pf >= PF_MID_LEVERAGE_THRESHOLD:
            active_leverage = base_leverage + 0.5 * (max_leverage - base_leverage)

    return max(1.0, float(active_leverage))


def current_trade_leverage(
    closes: Sequence[float],
    long_state: Sequence[Any],
    base_leverage: float = 1.0,
    max_leverage: float = 1.0,
    use_dynamic_leverage: bool = False,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
) -> float:
    """
    Reproduce la simulación de BacktestEngine._simulate sobre el histórico y devuelve el
    apalancamiento fijado en la entrada del trade actualmente abierto. El valor se decide
    al entrar y no cambia mientras el trade siga abierto, igual que en el backtest.
    Si no hay posición abierta devuelve el apalancamiento que aplicaría la próxima entrada.
    """
    max_leverage = max(base_leverage, max_leverage)
    cash = 1.0
    shares = 0.0
    entry_price = 0.0
    in_position = False
    trade_leverage = base_leverage
    closed_pnls = []

    for price, is_long in zip(closes, long_state):
        price = float(price)
        if is_long and not in_position:
            active_leverage = dynamic_leverage_from_pnls(closed_pnls, base_leverage, max_leverage, use_dynamic_leverage)
            adj_price = price * (1.0 + slippage_pct)
            investable = cash * active_leverage
            if investable > 0 and adj_price > 0:
                shares = investable / adj_price
                cash -= shares * adj_price
                entry_price = adj_price
                trade_leverage = active_leverage
                in_position = True
        elif not is_long and in_position:
            adj_price = price * (1.0 - slippage_pct)
            proceeds = shares * adj_price
            cash += proceeds
            closed_pnls.append(proceeds - (shares * entry_price))
            shares = 0.0
            in_position = False

    if in_position:
        return float(trade_leverage)
    return dynamic_leverage_from_pnls(closed_pnls, base_leverage, max_leverage, use_dynamic_leverage)
