from datetime import date
from decimal import Decimal
from typing import Tuple, Dict, Any
from apps.portfolios.models import Portfolio, Allocation
from apps.strategies.models import Strategy
from apps.accounting.models import LogicalPosition
from apps.marketdata.models import LatestQuote
from core.market_time import is_last_business_day


def evaluate_strategy_rebalance_trigger(
    portfolio: Portfolio,
    allocation: Allocation,
    as_of_date: date,
    signal_changed: bool = False,
    allocation_changed: bool = False,
) -> Tuple[bool, str, Decimal]:
    """
    Evaluates rebalancing trigger priority according to MVP section 6.2:
    Priority 1: Allocation change pending (effective_at <= now).
    Priority 2: Strategy signal changed.
    Priority 3: Periodic rebalance on last business day if drift > tolerance.
    Priority 4: Hold position.
    
    Returns: (should_rebalance: bool, reason: str, max_drift: Decimal)
    """
    strategy: Strategy = allocation.strategy

    # Priority 1: Allocation Change
    if allocation_changed:
        return True, "allocation_change", Decimal("1.0")

    # Priority 2: Signal Change
    if signal_changed:
        return True, "signal_change", Decimal("1.0")

    # Priority 3: Periodic Rebalance (daily / weekly / monthly / quarterly)
    freq = portfolio.rebalance_frequency
    tolerance = portfolio.rebalance_tolerance
    mode = portfolio.rebalance_tolerance_mode

    if freq == "daily" or (freq in ("weekly", "monthly", "quarterly") and is_last_business_day(as_of_date, frequency=freq)):
        # Calculate sleeve drift
        drift = calculate_sleeve_drift(portfolio, allocation)
        if drift > tolerance:
            return True, f"periodic_{freq}_drift_exceeded", drift

    # Priority 4: Hold
    return False, "hold_within_tolerance", Decimal("0.0")


def calculate_sleeve_drift(portfolio: Portfolio, allocation: Allocation) -> Decimal:
    """
    Calculates drift = |current_value - target_value| / target_value
    """
    # Sum current value of positions for this allocation's strategy
    positions = LogicalPosition.objects.filter(
        portfolio=portfolio,
        strategy=allocation.strategy,
    )
    current_sleeve_val = Decimal("0.0")
    for pos in positions:
        lq = LatestQuote.objects.filter(instrument=pos.instrument).first()
        price = lq.price if lq else Decimal("100.00")
        current_sleeve_val += pos.qty * price

    # Base equity estimate
    base_equity = Decimal("100000.00")
    if portfolio.kind == "paper":
        base_equity = portfolio.paper_initial_capital

    target_sleeve_val = base_equity * allocation.target_weight
    if target_sleeve_val <= Decimal("0.0001"):
        return Decimal("1.0") if current_sleeve_val > 0 else Decimal("0.0")

    drift = abs(current_sleeve_val - target_sleeve_val) / target_sleeve_val
    return drift
