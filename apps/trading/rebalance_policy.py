from datetime import date
from decimal import Decimal
from typing import Tuple
from apps.portfolios.models import Portfolio
from core.market_time import is_last_business_day


def is_periodic_rebalance_date(portfolio: Portfolio, as_of_date: date) -> bool:
    """
    True if `as_of_date` is a periodic rebalance date for the portfolio:
    daily -> every trading day; weekly / monthly / quarterly -> last business day of the period;
    never -> positions are only resized by signal or allocation changes.
    """
    freq = portfolio.rebalance_frequency
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


def decide_position_trade(
    portfolio: Portfolio,
    current_qty: Decimal,
    target_qty: Decimal,
    price: Decimal,
    equity: Decimal,
    is_rebalance_date: bool,
    allocation_changed: bool = False,
    force_rebalance: bool = False,
) -> Tuple[bool, str]:
    """
    Decides whether a sleeve position must be traded to its target on this run.
      1. Strategy entry / exit (flat <-> invested): always, every day.
      2. Manual rebalance requested or allocation weights changed: resize now.
      3. Periodic rebalance date and drift above the portfolio tolerance: resize.
      4. Otherwise hold the current quantity.

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

    if not is_rebalance_date:
        return False, "hold_until_rebalance_date"

    drift = position_drift(current_qty * price, target_qty * price, equity, portfolio.rebalance_tolerance_mode)
    if drift > portfolio.rebalance_tolerance:
        return True, f"periodic_{portfolio.rebalance_frequency}_drift_exceeded"
    return False, "hold_within_tolerance"
