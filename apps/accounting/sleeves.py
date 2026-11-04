"""
Contabilidad por manga (bot) dentro de un portafolio.

    sleeve_cash  = sum(SleeveCashEntry.amount)
    sleeve_value = sleeve_cash + valor de mercado de las LogicalPosition de la manga

Invariante: sum(sleeve_value) + caja no asignada = equity. La caja no asignada se deriva
(`equity - sum(sleeve_value)`); con equity del broker incluye intereses, dividendos no atribuidos
y comisiones externas (el "residual"), que se expone en `sleeve_diagnostics` y se absorbe en el
siguiente reajuste de portafolio.
"""
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Dict, Optional

from django.db.models import Sum

from apps.accounting.models import LogicalPosition, SleeveCashEntry
from apps.accounting.services import get_current_prices

ZERO = Decimal("0")


def _today() -> date:
    from core.market_time import get_ny_now
    return get_ny_now().date()


def add_entry(portfolio, strategy_id: int, entry_type: str, amount: Decimal,
              effective_date: Optional[date] = None, reference: str = "") -> Optional[SleeveCashEntry]:
    """Inserta un asiento de caja de manga (ignora importes nulos)."""
    if amount == 0:
        return None
    return SleeveCashEntry.objects.create(
        portfolio=portfolio, strategy_id=strategy_id, entry_type=entry_type, amount=amount,
        effective_date=effective_date or _today(), reference=reference,
    )


def sleeve_cash(portfolio) -> Dict[int, Decimal]:
    rows = SleeveCashEntry.objects.filter(portfolio=portfolio).values("strategy_id").annotate(t=Sum("amount"))
    return {r["strategy_id"]: r["t"] or ZERO for r in rows}


def positions_value_by_sleeve(portfolio, prices: Optional[Dict[int, dict]] = None) -> Dict[int, Decimal]:
    positions = list(LogicalPosition.objects.filter(portfolio=portfolio).exclude(qty=0).select_related("instrument"))
    if prices is None:
        prices = get_current_prices({p.instrument for p in positions}, refresh=False)
    out: Dict[int, Decimal] = defaultdict(lambda: ZERO)
    for p in positions:
        quote = prices.get(p.instrument_id)
        price = quote["price"] if quote else p.avg_cost
        out[p.strategy_id] += p.qty * price
    return dict(out)


def sleeve_values(portfolio, prices: Optional[Dict[int, dict]] = None) -> Dict[int, dict]:
    """{strategy_id: {"cash", "positions_value", "value"}} para toda manga con caja o posiciones."""
    cash = sleeve_cash(portfolio)
    pos = positions_value_by_sleeve(portfolio, prices)
    out = {}
    for sid in set(cash) | set(pos):
        c, pv = cash.get(sid, ZERO), pos.get(sid, ZERO)
        out[sid] = {"cash": c, "positions_value": pv, "value": c + pv}
    return out


def has_sleeve_accounts(portfolio) -> bool:
    return SleeveCashEntry.objects.filter(portfolio=portfolio).exists()


def ensure_sleeve_accounts(portfolio, equity: Optional[Decimal] = None) -> bool:
    """
    Inicializa la contabilidad por manga de un portafolio existente (idempotente): equivale a un
    reajuste en la fecha de migracion, dejando sleeve_value = equity x target_weight.
    Devuelve True si creo asientos.
    """
    if has_sleeve_accounts(portfolio):
        return False
    allocations = list(portfolio.allocations.filter(enabled=True))
    if not allocations:
        return False
    if equity is None or equity <= 0:
        from apps.accounting.services import portfolio_equity
        equity = portfolio_equity(portfolio)["equity"]
    if equity <= 0:
        return False
    pos = positions_value_by_sleeve(portfolio)
    for alloc in allocations:
        add_entry(portfolio, alloc.strategy_id, "initial",
                  equity * alloc.target_weight - pos.get(alloc.strategy_id, ZERO), reference="migration")
    return True


def record_fill_entries(portfolio, strategy_id: int, signed_qty: Decimal, price: Decimal,
                        commission: Decimal, reference: str) -> None:
    """Compra: caja -qty*precio; venta: +qty*precio; comision como `fee`."""
    add_entry(portfolio, strategy_id, "trade", -(signed_qty * price), reference=reference)
    if commission:
        add_entry(portfolio, strategy_id, "fee", -commission, reference=reference)


def allocate_cash_flow(portfolio, amount: Decimal, effective_date: Optional[date] = None, reference: str = "") -> None:
    """
    Reparte un aporte externo entre mangas: depositos segun target_weight (el resto queda sin asignar),
    retiros pro-rata al sleeve_value actual.
    """
    if amount == 0 or not has_sleeve_accounts(portfolio):
        return
    if amount > 0:
        for alloc in portfolio.allocations.filter(enabled=True):
            add_entry(portfolio, alloc.strategy_id, "deposit_alloc", amount * alloc.target_weight,
                      effective_date, reference)
        return
    enabled = set(portfolio.allocations.filter(enabled=True).values_list("strategy_id", flat=True))
    values = {sid: v["value"] for sid, v in sleeve_values(portfolio).items() if sid in enabled}
    total = sum((v for v in values.values() if v > 0), ZERO)
    if total <= 0:
        return
    for sid, v in values.items():
        if v > 0:
            add_entry(portfolio, sid, "withdrawal_alloc", amount * (v / total), effective_date, reference)


def release_disabled_sleeves(portfolio, reference: str = "disabled_sleeve") -> None:
    """La caja de una manga deshabilitada y sin posiciones vuelve a 'no asignada'."""
    enabled = set(portfolio.allocations.filter(enabled=True).values_list("strategy_id", flat=True))
    for sid, v in sleeve_values(portfolio).items():
        if sid not in enabled and v["positions_value"] == 0 and v["cash"] != 0:
            add_entry(portfolio, sid, "transfer", -v["cash"], reference=reference)


def sleeve_diagnostics(portfolio, equity: Decimal) -> dict:
    """Estado de las mangas y el residual contra el equity (para diagnostico, no se esconde)."""
    enabled = {a.strategy_id: a for a in portfolio.allocations.filter(enabled=True)}
    values = sleeve_values(portfolio)
    empty = {"cash": ZERO, "positions_value": ZERO, "value": ZERO}
    total = sum((v["value"] for sid, v in values.items() if sid in enabled), ZERO)
    return {
        "equity": equity,
        "sleeves": {
            sid: {
                **values.get(sid, empty),
                "weight_actual": (values.get(sid, empty)["value"] / equity) if equity > 0 else ZERO,
                "weight_target": a.target_weight,
            }
            for sid, a in enabled.items()
        },
        "sleeves_total": total,
        "unassigned": equity - total,
    }


def external_flows(portfolio, strategy_id: int, after: Optional[date], upto: date) -> Decimal:
    """Capital que entro/salio de la manga sin ser rendimiento (reparto inicial, transferencias, aportes)."""
    qs = SleeveCashEntry.objects.filter(
        portfolio=portfolio, strategy_id=strategy_id,
        entry_type__in=SleeveCashEntry.EXTERNAL_TYPES, effective_date__lte=upto,
    )
    if after:
        qs = qs.filter(effective_date__gt=after)
    return qs.aggregate(t=Sum("amount"))["t"] or ZERO
