"""
Cash sweep: opcion por portafolio para invertir en un ETF de letras del Tesoro (SGOV por defecto)
el efectivo que ningun bot esta usando.

El sweep es una estrategia de sistema (`kind="cash_sweep"`, sin `Allocation`): su posicion es una
`LogicalPosition` normal, asi que reutiliza fills, conciliacion y valoracion. El objetivo se calcula
a nivel portafolio, despues de los targets de todos los bots:

    long_mv            = suma(cantidad final x precio) de las posiciones de los bots
    free               = equity - long_mv - buffer_pct x equity
    target_sweep_value = max(0, free)            (0 si long_mv > equity: nunca margen para tener T-bills)
"""
from dataclasses import dataclass
from decimal import Decimal
from typing import List

from apps.instruments.models import Instrument
from apps.strategies.models import Strategy
from core.money import quantize_qty

SWEEP_KIND = "cash_sweep"
SWEEP_SLUG = "cash-sweep"
SWEEP_SYMBOLS = ("SGOV", "BIL", "USFR")
DEFAULT_SWEEP_SYMBOL = "SGOV"

ZERO = Decimal("0")


def get_sweep_strategy() -> Strategy:
    """Estrategia de sistema unica del cash sweep (se crea la primera vez que hace falta)."""
    strategy, _ = Strategy.objects.get_or_create(
        slug=SWEEP_SLUG,
        defaults={
            "name": "Cash sweep", "family": "System", "kind": SWEEP_KIND, "engine": SWEEP_KIND,
            "is_active": True, "signal_only": False,
        },
    )
    return strategy


def ensure_sweep_instrument(symbol: str) -> Instrument:
    inst, _ = Instrument.objects.get_or_create(symbol=symbol, defaults={"name": symbol, "asset_class": "etf"})
    return inst


def sweep_instrument_ids() -> set:
    """Instrumentos del sweep de los portafolios que lo tienen activo (entran al universo de marketdata)."""
    from apps.portfolios.models import Portfolio
    symbols = set(
        Portfolio.objects.filter(cash_sweep_enabled=True, status="active").values_list("cash_sweep_symbol", flat=True)
    )
    return set(Instrument.objects.filter(symbol__in=symbols).values_list("id", flat=True))


@dataclass(frozen=True)
class SweepPlan:
    target_qty: Decimal
    delta_qty: Decimal
    reason: str          # cash_sweep_in | cash_sweep_out | cash_sweep_hold
    target_value: Decimal


def plan_sweep(
    *, enabled: bool, equity: Decimal, long_mv: Decimal, buffer_pct: Decimal, min_trade_usd: Decimal,
    price: Decimal, actual_qty: Decimal, bot_buys_value: Decimal,
) -> SweepPlan:
    """
    Objetivo del sweep y decision de operar.
      - Desactivado, o portafolio usando margen (long_mv > equity): objetivo 0 y se liquida siempre.
      - Ajuste menor a `min_trade_usd`: no se opera, salvo que haga falta caja para compras de bots
        (el sweep debe vender) o haya que liquidarlo.
    """
    forced = False
    if not enabled or long_mv > equity or price <= 0:
        target_value, forced = ZERO, True
    else:
        target_value = max(ZERO, equity - long_mv - buffer_pct * equity)

    target_qty = quantize_qty(target_value / price, round_down=True) if price > 0 else ZERO
    delta = target_qty - actual_qty
    if delta == 0:
        return SweepPlan(actual_qty, ZERO, "cash_sweep_hold", target_value)

    if delta < 0 and (forced or bot_buys_value > 0):
        return SweepPlan(target_qty, delta, "cash_sweep_out", target_value)
    if abs(delta) * price < min_trade_usd:
        return SweepPlan(actual_qty, ZERO, "cash_sweep_hold", target_value)
    return SweepPlan(target_qty, delta, "cash_sweep_in" if delta > 0 else "cash_sweep_out", target_value)


def order_intent_ids_sells_first(run) -> List[int]:
    """
    Ids de los intents pendientes del run: primero las ventas (las del sweep antes que las demas, para
    liberar caja), luego las compras. Las ordenes son de mercado DAY y se despachan en secuencia.
    """
    sweep_inst_ids = set(
        run.targets.filter(strategy__kind=SWEEP_KIND).values_list("instrument_id", flat=True)
    )
    intents = list(run.intents.filter(status="pending"))
    intents.sort(key=lambda i: (
        0 if i.net_delta_qty < 0 else 1,
        0 if i.instrument_id in sweep_inst_ids else 1,
        i.id,
    ))
    return [i.id for i in intents]
