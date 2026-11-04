"""Metricas del cash sweep de un portafolio (saldo, rendimiento en USD, anualizado y contribucion)."""
from decimal import Decimal
from typing import Optional

from apps.accounting.models import EquitySnapshot, SleevePnlDaily
from apps.accounting.sleeves import positions_value_by_sleeve
from apps.strategies.models import Strategy

ZERO = Decimal("0")
TRADING_DAYS = 252


def cash_sweep_metrics(portfolio) -> dict:
    """
    - current_balance: valor de mercado actual en T-bills y saldo promedio del periodo.
    - cumulative_pnl_usd: suma de los pnl diarios del sweep (variacion de precio + dividendos netos
      - costes), es decir, rendimiento total.
    - annualized_return_pct = suma(pnl) / saldo promedio x 252 / dias.
    - contribution_bps: suma(pnl) / equity al inicio del periodo del sweep, en puntos basicos.
    """
    sweep = Strategy.objects.filter(kind="cash_sweep").first()
    base = {
        "enabled": portfolio.cash_sweep_enabled, "symbol": portfolio.cash_sweep_symbol,
        "current_balance": 0.0, "average_balance": 0.0, "cumulative_pnl_usd": 0.0,
        "annualized_return_pct": None, "contribution_bps": None, "days": 0, "since": None,
    }
    if not sweep:
        return base

    rows = list(SleevePnlDaily.objects.filter(portfolio=portfolio, strategy=sweep).order_by("date"))
    current = positions_value_by_sleeve(portfolio).get(sweep.id, ZERO)
    base["current_balance"] = float(round(current, 2))
    if not rows:
        return base

    total_pnl = sum((r.pnl for r in rows), ZERO)
    days = len(rows)
    average: Decimal = sum((r.positions_value for r in rows), ZERO) / days
    start_equity: Optional[Decimal] = (
        EquitySnapshot.objects.filter(portfolio=portfolio, date__lte=rows[0].date)
        .order_by("-date").values_list("total_equity", flat=True).first()
    )
    return {
        **base,
        "average_balance": float(round(average, 2)),
        "cumulative_pnl_usd": float(round(total_pnl, 2)),
        "annualized_return_pct": float(round(total_pnl / average * TRADING_DAYS / days * 100, 2)) if average > 0 else None,
        "contribution_bps": float(round(total_pnl / start_equity * 10000, 1)) if start_equity and start_equity > 0 else None,
        "days": days,
        "since": rows[0].date.isoformat(),
    }
