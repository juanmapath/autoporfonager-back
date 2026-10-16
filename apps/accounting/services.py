"""
Valoración de portafolios para la interfaz: precios actuales, equity, posiciones abiertas
con PnL no realizado y apalancamiento, e historial de equity frente a índices de referencia.
"""
import logging
import math
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, Iterable, Optional

from django.db.models import Sum
from django.utils import timezone

from apps.accounting.models import LogicalPosition, EquitySnapshot, LedgerEntry
from apps.instruments.models import Instrument
from apps.marketdata.models import Bar, LatestQuote

logger = logging.getLogger(__name__)

# Una cotización se vuelve a pedir al proveedor si se descargó hace más de esto.
QUOTE_MAX_AGE = timedelta(minutes=15)
# Las barras de los índices se refrescan como mucho una vez cada este intervalo.
BENCHMARK_REFRESH_AGE = timedelta(hours=6)

BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000",
}


def get_current_prices(instruments: Iterable[Instrument], refresh: bool = True) -> Dict[int, dict]:
    """
    Devuelve {instrument_id: {"price", "as_of", "source"}} para cada instrumento.
    Con `refresh`, las cotizaciones de más de QUOTE_MAX_AGE se vuelven a descargar; si la
    descarga falla se usa la última conocida y, sin cotización, el último cierre diario.
    """
    from apps.marketdata.services import MarketDataService

    svc = None
    prices = {}
    now = timezone.now()
    for inst in instruments:
        quote = LatestQuote.objects.filter(instrument=inst).first()
        if refresh and (quote is None or now - quote.updated_at > QUOTE_MAX_AGE):
            svc = svc or MarketDataService()
            quote = svc.update_latest_quote(inst.symbol) or quote
        if quote:
            prices[inst.id] = {"price": quote.price, "as_of": quote.timestamp, "source": "quote"}
            continue
        bar = Bar.objects.filter(instrument=inst).order_by("-date").first()
        if bar:
            prices[inst.id] = {"price": bar.close, "as_of": bar.date, "source": "close"}
    return prices


def _book_equity(portfolio, unrealized_pnl: Decimal) -> Decimal:
    """Equity contable: capital inicial + PnL realizado + PnL no realizado - comisiones."""
    from apps.trading.models import Fill

    realized = LogicalPosition.objects.filter(portfolio=portfolio).aggregate(t=Sum("realized_pnl"))["t"] or Decimal("0")
    fees = Fill.objects.filter(order__portfolio=portfolio).aggregate(t=Sum("commission"))["t"] or Decimal("0")
    return portfolio.paper_initial_capital + realized + unrealized_pnl - fees


def portfolio_equity(portfolio, unrealized_pnl: Optional[Decimal] = None, refresh: bool = False) -> dict:
    """
    Equity actual del portafolio. Con broker sincronizado manda el saldo del broker;
    si no, se calcula desde la contabilidad interna.
    """
    broker = portfolio.broker_accounts.filter(trading_enabled=True).first()
    if broker and broker.managed_capital:
        return {
            "equity": broker.managed_capital,
            "source": "broker",
            "synced_at": broker.updated_at,
        }
    if unrealized_pnl is None:
        unrealized_pnl = open_positions(portfolio, refresh=refresh)["summary"]["unrealized_pnl"]
    return {
        "equity": _book_equity(portfolio, Decimal(str(unrealized_pnl))),
        "source": "book",
        "synced_at": None,
    }


def open_positions(portfolio, refresh: bool = True) -> dict:
    """
    Posiciones abiertas valoradas a precio actual, con PnL no realizado y desglose de
    capital propio frente a capital financiado.

    - Por manga (estrategia): apalancamiento = valor de mercado / capital asignado a la manga.
      El valor de cada posición se reparte en propio (valor / apalancamiento) y financiado.
    - Por portafolio: el margen sólo existe si la exposición bruta supera el equity.
    """
    positions = list(
        LogicalPosition.objects.filter(portfolio=portfolio)
        .exclude(qty=0)
        .select_related("strategy", "instrument")
    )
    prices = get_current_prices({p.instrument for p in positions}, refresh=refresh)

    rows = []
    unrealized_total = Decimal("0")
    for p in positions:
        quote = prices.get(p.instrument_id)
        price = quote["price"] if quote else p.avg_cost
        market_value = p.qty * price
        cost_basis = p.qty * p.avg_cost
        unrealized = market_value - cost_basis
        unrealized_total += unrealized
        rows.append({
            "strategy_id": p.strategy_id,
            "strategy": p.strategy.name,
            "instrument": p.instrument.symbol,
            "qty": p.qty,
            "avg_cost": p.avg_cost,
            "last_price": price,
            "price_as_of": quote["as_of"] if quote else None,
            "price_source": quote["source"] if quote else "avg_cost",
            "market_value": market_value,
            "cost_basis": cost_basis,
            "unrealized_pnl": unrealized,
            "unrealized_pnl_pct": (unrealized / cost_basis * 100) if cost_basis else Decimal("0"),
        })

    eq = portfolio_equity(portfolio, unrealized_pnl=unrealized_total)
    equity = eq["equity"]

    weights = {
        a.strategy_id: a.target_weight
        for a in portfolio.allocations.filter(enabled=True)
    }
    sleeve_value: Dict[int, Decimal] = {}
    for r in rows:
        sleeve_value[r["strategy_id"]] = sleeve_value.get(r["strategy_id"], Decimal("0")) + abs(r["market_value"])

    for r in rows:
        weight = weights.get(r["strategy_id"])
        sleeve_capital = equity * weight if weight else None
        leverage = (sleeve_value[r["strategy_id"]] / sleeve_capital) if sleeve_capital else None
        own = abs(r["market_value"]) / max(leverage, Decimal("1")) if leverage else abs(r["market_value"])
        r["sleeve_capital"] = sleeve_capital
        r["sleeve_leverage"] = leverage
        r["own_capital"] = own
        r["financed"] = abs(r["market_value"]) - own

    gross = sum((abs(r["market_value"]) for r in rows), Decimal("0"))
    summary = {
        "equity": equity,
        "equity_source": eq["source"],
        "equity_synced_at": eq["synced_at"],
        "gross_exposure": gross,
        "own_capital_invested": min(gross, equity) if equity > 0 else gross,
        "financed": max(Decimal("0"), gross - equity),
        "cash": equity - gross,
        "leverage": (gross / equity) if equity > 0 else None,
        "unrealized_pnl": unrealized_total,
    }
    return {"positions": rows, "summary": summary}


def ensure_benchmark_bars(symbol: str, start: date) -> Optional[Instrument]:
    """
    Garantiza barras diarias del índice desde `start`. Los índices se crean como no operables
    para que el pipeline diario no los trate como activos de estrategias.
    """
    from apps.marketdata.services import MarketDataService

    instrument, _ = Instrument.objects.get_or_create(
        symbol=symbol, defaults={"name": BENCHMARKS.get(symbol, symbol), "asset_class": "etf", "tradable": False},
    )
    bars = Bar.objects.filter(instrument=instrument)
    first = bars.order_by("date").first()
    latest_update = bars.order_by("-updated_at").values_list("updated_at", flat=True).first()

    days = None
    if first is None or first.date > start:
        days = (date.today() - start).days + 10
    elif latest_update is None or timezone.now() - latest_update > BENCHMARK_REFRESH_AGE:
        days = 10
    if days:
        MarketDataService().sync_bars(symbol, days=days)
    return instrument


CASH_FLOW_TYPES = ("deposit", "withdrawal")


def net_cash_flows(portfolio) -> Decimal:
    """Depósitos menos retiros registrados desde la vinculación."""
    return (
        LedgerEntry.objects.filter(portfolio=portfolio, entry_type__in=CASH_FLOW_TYPES)
        .aggregate(t=Sum("amount"))["t"] or Decimal("0")
    )


def equity_series(portfolio) -> list:
    """
    Serie diaria del portafolio: equity, flujo externo neto del día, capital aportado acumulado
    y retorno logarítmico diario neutralizado de flujos:

        r_t = ln((E_t - F_t) / E_{t-1})

    donde F_t son los depósitos (+) y retiros (-) con fecha en (t-1, t]. Así un depósito no
    cuenta como ganancia ni un retiro como pérdida, y la suma de r_t da el rendimiento
    ponderado en el tiempo (TWR): exp(Σ r_t) - 1.
    """
    points = [
        {"date": s.date, "equity": s.total_equity}
        for s in EquitySnapshot.objects.filter(portfolio=portfolio).order_by("date")
    ]
    from core.market_time import get_ny_now
    today = get_ny_now().date()
    if not points or points[-1]["date"] < today:
        points.append({"date": today, "equity": open_positions(portfolio, refresh=False)["summary"]["equity"]})

    flows = list(
        LedgerEntry.objects.filter(portfolio=portfolio, entry_type__in=CASH_FLOW_TYPES)
        .exclude(effective_date=None)
        .values_list("effective_date", "amount")
    )

    contributed = portfolio.paper_initial_capital + sum((a for d, a in flows if d <= points[0]["date"]), Decimal("0"))
    prev = None
    for p in points:
        flow = Decimal("0")
        if prev is not None:
            flow = sum((a for d, a in flows if prev["date"] < d <= p["date"]), Decimal("0"))
        contributed += flow
        log_ret = 0.0
        if prev is not None and prev["equity"] > 0 and p["equity"] - flow > 0:
            log_ret = math.log(float((p["equity"] - flow) / prev["equity"]))
        p.update({"net_flow": flow, "contributed": contributed, "log_return": log_ret})
        prev = p
    return points


def time_weighted_return(points: list) -> float:
    """Rendimiento acumulado (fracción) a partir de los retornos logarítmicos diarios."""
    return math.exp(sum(p["log_return"] for p in points)) - 1


def equity_history(portfolio, benchmark_symbols: Iterable[str]) -> dict:
    """Serie diaria del portafolio (ver `equity_series`) y cierres ajustados de los índices pedidos."""
    series = equity_series(portfolio)

    benchmarks = {}
    start = series[0]["date"] - timedelta(days=7)
    for symbol in benchmark_symbols:
        if symbol not in BENCHMARKS:
            continue
        try:
            instrument = ensure_benchmark_bars(symbol, start)
        except Exception as e:
            logger.error(f"No se pudieron cargar las barras del índice {symbol}: {e}")
            continue
        bars = Bar.objects.filter(instrument=instrument, date__gte=start).order_by("date").values("date", "adj_close")
        benchmarks[symbol] = {
            "name": BENCHMARKS[symbol],
            "series": [{"date": b["date"], "close": b["adj_close"]} for b in bars],
        }

    return {"portfolio": series, "benchmarks": benchmarks}
