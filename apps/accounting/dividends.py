"""
Dividendos: ledger (`dividend` / `dividend_withholding`) + caja de la manga duena de la posicion.

Un dividendo es rendimiento, no un aporte externo: NO forma parte de CASH_FLOW_TYPES, asi que el TWR
de `equity_series` lo cuenta como ganancia. En paper suma al equity contable (`_book_equity`); con broker
el equity ya lo incluye y estas entradas son informativas + atribucion por manga.
"""
import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from django.db.models import Sum

from apps.accounting.models import LedgerEntry, LogicalPosition
from apps.accounting.sleeves import add_entry, has_sleeve_accounts
from core.money import quantize_amount

logger = logging.getLogger(__name__)

PAPER_DIVIDEND_LOOKBACK_DAYS = 7


def record_dividend(portfolio, strategy_id: Optional[int], amount: Decimal, effective_date: date,
                    reference: str, withholding: bool = False, broker_account=None) -> bool:
    """Crea el asiento de ledger y, si hay manga duena, su `SleeveCashEntry` de tipo dividend. Idempotente por `reference`."""
    if LedgerEntry.objects.filter(portfolio=portfolio, reference=reference).exists():
        return False
    amount = quantize_amount(amount)
    if amount == 0:
        return False
    LedgerEntry.objects.create(
        portfolio=portfolio, broker_account=broker_account,
        entry_type="dividend_withholding" if withholding else "dividend",
        amount=amount, reference=reference, effective_date=effective_date,
    )
    if strategy_id and has_sleeve_accounts(portfolio):
        add_entry(portfolio, strategy_id, "dividend", amount, effective_date, reference)
    return True


def qty_held_on(portfolio, position: LogicalPosition, ex_date: date) -> Decimal:
    """Cantidad que la manga tenia el dia ex-dividendo: la actual menos lo operado desde esa fecha."""
    from apps.trading.models import FillAllocation

    after = FillAllocation.objects.filter(
        fill__order__portfolio=portfolio, strategy_id=position.strategy_id,
        fill__order__instrument_id=position.instrument_id, fill__filled_at__date__gte=ex_date,
    ).aggregate(t=Sum("allocated_qty"))["t"] or Decimal("0")
    return position.qty - after


def accrue_paper_dividends(provider=None, today: Optional[date] = None,
                           lookback_days: int = PAPER_DIVIDEND_LOOKBACK_DAYS) -> Dict[str, int]:
    """
    Portafolios paper: por cada posicion con cantidad > 0 el dia ex-dividendo se acredita
    cantidad x dividendo. Yahoo solo informa la fecha ex-dividendo, que se usa como fecha efectiva.
    """
    from apps.marketdata.providers.yahoo import YahooProvider
    from apps.portfolios.models import Portfolio
    from core.market_time import get_ny_now

    provider = provider or YahooProvider()
    today = today or get_ny_now().date()
    start = today - timedelta(days=lookback_days)
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)

    created = errors = 0
    dividends_by_symbol: Dict[str, List[dict]] = {}
    for portfolio in Portfolio.objects.filter(kind="paper", status="active"):
        positions = LogicalPosition.objects.filter(portfolio=portfolio, qty__gt=0).select_related("instrument")
        for pos in positions:
            inst = pos.instrument
            if inst.symbol not in dividends_by_symbol:
                try:
                    dividends_by_symbol[inst.symbol] = provider.get_dividends(inst.yahoo_symbol or inst.symbol, start_dt)
                except Exception as e:
                    logger.error(f"No se pudieron leer los dividendos de {inst.symbol}: {e}")
                    dividends_by_symbol[inst.symbol] = []
                    errors += 1
            for div in dividends_by_symbol[inst.symbol]:
                if not (start <= div["ex_date"] <= today):
                    continue
                qty = qty_held_on(portfolio, pos, div["ex_date"])
                if qty <= 0:
                    continue
                ref = f"div:{portfolio.id}:{pos.strategy_id}:{inst.symbol}:{div['ex_date'].isoformat()}"
                if record_dividend(portfolio, pos.strategy_id, qty * Decimal(str(div["amount"])), div["ex_date"], ref):
                    created += 1
    return {"created": created, "errors": errors}
