from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple
import pandas as pd

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

DECISION_OFFSET_MINUTES = 25
MOC_CUTOFF_OFFSET_MINUTES = 12


def get_ny_now() -> datetime:
    return datetime.now(NY_TZ)


def is_trading_day(dt: Optional[date] = None) -> bool:
    """Checks whether the date is a business trading day (NYSE)."""
    check_date = dt or get_ny_now().date()
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("XNYS")
        schedule = nyse.schedule(start_date=check_date, end_date=check_date)
        return not schedule.empty
    except Exception:
        # Fallback: Monday-Friday
        return check_date.weekday() < 5


def get_market_hours(dt: Optional[date] = None) -> Optional[Tuple[datetime, datetime]]:
    """Returns (market_open, market_close) in NY_TZ or None if market is closed."""
    check_date = dt or get_ny_now().date()
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("XNYS")
        schedule = nyse.schedule(start_date=check_date, end_date=check_date)
        if schedule.empty:
            return None
        market_open = schedule.iloc[0]["market_open"].to_pydatetime().astimezone(NY_TZ)
        market_close = schedule.iloc[0]["market_close"].to_pydatetime().astimezone(NY_TZ)
        return market_open, market_close
    except Exception:
        if check_date.weekday() >= 5:
            return None
        market_open = datetime.combine(check_date, time(9, 30), tzinfo=NY_TZ)
        market_close = datetime.combine(check_date, time(16, 0), tzinfo=NY_TZ)
        return market_open, market_close


def get_decision_window(dt: Optional[date] = None) -> Optional[Tuple[datetime, datetime]]:
    """Returns (decision_start, moc_cutoff) for market orders on given day."""
    hours = get_market_hours(dt)
    if not hours:
        return None
    _, market_close = hours
    decision_start = market_close - timedelta(minutes=DECISION_OFFSET_MINUTES)
    moc_cutoff = market_close - timedelta(minutes=MOC_CUTOFF_OFFSET_MINUTES)
    return decision_start, moc_cutoff


def is_last_business_day(dt: Optional[date] = None, frequency: str = "monthly") -> bool:
    """
    Evaluates whether the given date is the last trading business day of the period:
    frequency in ['daily', 'weekly', 'monthly', 'quarterly'].
    """
    check_date = dt or get_ny_now().date()
    if not is_trading_day(check_date):
        return False
    
    if frequency == "daily":
        return True

    # Scan the next days in the current calendar week/month/quarter to see if another trading day exists
    if frequency == "weekly":
        # Check if any subsequent day in this week (up to Sunday) is a trading day
        current_weekday = check_date.weekday()
        days_left_in_week = 6 - current_weekday
        for d in range(1, days_left_in_week + 1):
            next_d = check_date + timedelta(days=d)
            if is_trading_day(next_d):
                return False
        return True

    if frequency == "monthly":
        # Check if any subsequent day in the current calendar month is a trading day
        next_d = check_date + timedelta(days=1)
        while next_d.month == check_date.month:
            if is_trading_day(next_d):
                return False
            next_d += timedelta(days=1)
        return True

    if frequency == "quarterly":
        # Quarters end in months 3, 6, 9, 12
        if check_date.month not in [3, 6, 9, 12]:
            return False
        return is_last_business_day(check_date, frequency="monthly")

    return False
