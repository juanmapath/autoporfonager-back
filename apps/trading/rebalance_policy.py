"""
Politica de rebalanceo en dos niveles.

Nivel 1 - Portafolio (capital entre bots): `Portfolio.rebalance_frequency / tolerance / mode`.
    En fecha de rebalanceo del portafolio, si alguna manga se desvia mas que la tolerancia se
    reajustan todas a `equity x target_weight` (transferencias de caja entre mangas).
Nivel 2 - Bot (capital entre los activos del bot): parametros de la version
    (`rebalance_frequency`, `rebalance_tolerance`, `rebalance_tolerance_mode`); si no estan
    definidos heredan los del portafolio.
"""
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional, Tuple

from apps.portfolios.models import Portfolio
from core.market_time import is_last_business_day

INHERIT_VALUES = (None, "", "inherit")


def is_portfolio_rebalance_date(portfolio: Portfolio, as_of_date: date) -> bool:
    """
    True si `as_of_date` es fecha de rebalanceo del portafolio:
    daily -> todos los dias de mercado; weekly / monthly / quarterly / quarterly_earnings -> ultimo dia
    habil del periodo; never -> solo por cambio de pesos o rebalanceo manual.
    """
    freq = portfolio.rebalance_frequency
    if freq == "never":
        return False
    return is_last_business_day(as_of_date, frequency=freq)


# Nombre anterior, conservado por compatibilidad.
is_periodic_rebalance_date = is_portfolio_rebalance_date


@dataclass(frozen=True)
class RebalanceConfig:
    frequency: str
    tolerance: Decimal
    mode: str


def bot_rebalance_config(version, portfolio: Portfolio) -> RebalanceConfig:
    """Frecuencia/tolerancia/modo efectivos del bot; lo no definido hereda del portafolio."""
    params = (getattr(version, "params", None) or {}) if version is not None else {}
    freq = params.get("rebalance_frequency")
    tol = params.get("rebalance_tolerance")
    mode = params.get("rebalance_tolerance_mode")

    if freq in INHERIT_VALUES:
        freq = portfolio.rebalance_frequency
    if mode in INHERIT_VALUES:
        mode = portfolio.rebalance_tolerance_mode
    if tol in INHERIT_VALUES:
        tolerance = Decimal(str(portfolio.rebalance_tolerance))
    else:
        try:
            tolerance = Decimal(str(tol))
        except (InvalidOperation, ValueError):
            tolerance = Decimal(str(portfolio.rebalance_tolerance))
    return RebalanceConfig(frequency=freq, tolerance=tolerance, mode=mode)


def is_bot_rebalance_date(version, portfolio: Portfolio, as_of_date: date) -> bool:
    """Fecha de rebalanceo del bot (su propia frecuencia, o la del portafolio si no la define)."""
    freq = bot_rebalance_config(version, portfolio).frequency
    if freq == "never":
        return False
    return is_last_business_day(as_of_date, frequency=freq)


def position_drift(current_value: Decimal, target_value: Decimal, equity: Decimal, mode: str) -> Decimal:
    """
    relative: |current - target| / target   (deviation from the position's own target)
    absolute: |current - target| / equity   (deviation in portfolio weight points)
    """
    gap = abs(current_value - target_value)
    base = target_value if mode == "relative" else equity
    if base <= 0:
        return Decimal("1.0") if gap > 0 else Decimal("0.0")
    return gap / base


def needs_portfolio_reset(
    portfolio: Portfolio,
    sleeve_values: Dict[int, Decimal],
    target_weights: Dict[int, Decimal],
    equity: Decimal,
    is_rebalance_date: bool,
    allocation_changed: bool = False,
    force_rebalance: bool = False,
    missing_sleeve: bool = False,
) -> Tuple[bool, str]:
    """
    Decide si hay que reajustar TODAS las mangas a equity x target_weight:
      1. rebalanceo manual pendiente           -> "manual_rebalance"
      2. cambio de pesos / manga nueva         -> "allocation_change"
      3. fecha del portafolio y alguna manga fuera de banda -> "portfolio_band_exceeded"
    """
    if force_rebalance:
        return True, "manual_rebalance"
    if allocation_changed or missing_sleeve:
        return True, "allocation_change"
    if not is_rebalance_date or equity <= 0:
        return False, "no_reset"

    for sid, target_w in target_weights.items():
        actual_w = sleeve_values.get(sid, Decimal("0")) / equity
        gap = abs(actual_w - target_w)
        if portfolio.rebalance_tolerance_mode == "relative":
            drift = (gap / target_w) if target_w > 0 else (Decimal("1") if gap > 0 else Decimal("0"))
        else:
            drift = gap
        if drift > portfolio.rebalance_tolerance:
            return True, "portfolio_band_exceeded"
    return False, "within_band"


def decide_position_trade(
    portfolio: Portfolio,
    current_qty: Decimal,
    target_qty: Decimal,
    price: Decimal,
    equity: Decimal,
    is_rebalance_date: bool,
    allocation_changed: bool = False,
    force_rebalance: bool = False,
    config: Optional[RebalanceConfig] = None,
    portfolio_reset: bool = False,
) -> Tuple[bool, str]:
    """
    Decides whether a sleeve position must be traded to its target on this run.
      1. Strategy entry / exit (flat <-> invested): always, every day.
      2. Manual rebalance requested, allocation weights changed or the sleeve was just reset
         by a portfolio rebalance: resize now.
      3. Bot rebalance date (`is_rebalance_date` refers to the BOT) and drift above the bot's
         tolerance: resize.
      4. Otherwise hold the current quantity.

    `config` is the bot's tolerance/mode (defaults to the portfolio's, as before).
    Returns: (should_trade, reason)
    """
    if current_qty == target_qty:
        return False, "no_change"
    if current_qty <= 0:
        return True, "signal_entry"
    if target_qty <= 0:
        return True, "signal_exit"

    if force_rebalance:
        return True, "manual_rebalance"
    if allocation_changed:
        return True, "allocation_change"
    if portfolio_reset:
        return True, "portfolio_reset"

    if not is_rebalance_date:
        return False, "hold_until_rebalance_date"

    tolerance = config.tolerance if config else portfolio.rebalance_tolerance
    mode = config.mode if config else portfolio.rebalance_tolerance_mode
    drift = position_drift(current_qty * price, target_qty * price, equity, mode)
    if drift > tolerance:
        return True, "bot_rebalance"
    return False, "hold_within_tolerance"
