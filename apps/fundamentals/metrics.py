"""
Metrics Catalog and Computation Engine.

This module provides:
1. Declarative definitions of metrics (direction, source, label, description).
2. Calculation logic for base & derived fundamentals (TTM rolling sums, YoY growth)
   and price-derived metrics (52w-low, momentum, volatility).
3. Ranking and percentile scoring with neutral (0.5) fallback for missing data.
"""
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Any
import numpy as np
import pandas as pd

from apps.instruments.models import Instrument
from apps.fundamentals.models import FundamentalStatement
from apps.marketdata.models import Bar, LatestQuote


@dataclass
class MetricContext:
    instrument: Instrument
    as_of: date
    price: Optional[float]
    bars_df: Optional[pd.DataFrame]  # OHLCV bars up to as_of
    statements: List[FundamentalStatement]  # Available statements up to as_of (period_type='Q' or 'FY')


@dataclass
class MetricDef:
    key: str
    label: str
    direction: str  # "asc" (menor es mejor, ej: PER) | "desc" (mayor es mejor, ej: ROIC)
    source: str     # "fundamental" | "price" | "mixed"
    description: str
    fn: Callable[[MetricContext], Optional[float]]


# ---------------------------------------------------------------------------
# Helper functions for statement math (TTM)
# ---------------------------------------------------------------------------
def _get_ttm_sum(statements: List[FundamentalStatement], field_name: str, min_qtrs: int = 4) -> Optional[float]:
    """Sums the last 4 quarterly statements for flow items (revenue, net_income, ebit, etc.)."""
    q_stmts = [s for s in statements if s.period_type == "Q"]
    if len(q_stmts) < min_qtrs:
        # Fallback: check if we have an FY statement
        fy_stmts = [s for s in statements if s.period_type == "FY"]
        if fy_stmts:
            return fy_stmts[0].data.get(field_name)
        return None

    vals = []
    for s in q_stmts[:4]:
        v = s.data.get(field_name)
        if v is None:
            return None
        vals.append(v)
    return float(sum(vals))


def _get_latest_balance_val(statements: List[FundamentalStatement], field_name: str) -> Optional[float]:
    """Retrieves the latest available balance sheet point-in-time value."""
    if not statements:
        return None
    return statements[0].data.get(field_name)


# ---------------------------------------------------------------------------
# Metric Calculation Callables
# ---------------------------------------------------------------------------
def calc_per(ctx: MetricContext) -> Optional[float]:
    """Price to Earnings Ratio (P/E). Lower is better (asc)."""
    if not ctx.price or ctx.price <= 0:
        return None
    eps = _get_ttm_sum(ctx.statements, "eps_diluted")
    if eps is None or eps <= 0:
        return None  # Negative/zero earnings treated as worst rank
    return float(ctx.price / eps)


def calc_peg(ctx: MetricContext) -> Optional[float]:
    """PEG Ratio: P/E divided by EPS YoY growth (%). Lower is better (asc)."""
    pe = calc_per(ctx)
    if pe is None:
        return None
    eps_growth = calc_eps_growth_yoy(ctx)
    if eps_growth is None or eps_growth <= 0:
        return None
    # peg = PE / (growth_pct * 100)
    return float(pe / (eps_growth * 100.0))


def calc_roic(ctx: MetricContext) -> Optional[float]:
    """
    Return on Invested Capital (ROIC). Higher is better (desc).
    ROIC = NOPAT / Invested Capital
    NOPAT approx = Operating Income * (1 - Tax Rate)
    Invested Capital = Total Debt + Total Equity - Cash
    """
    op_inc = _get_ttm_sum(ctx.statements, "operating_income")
    if op_inc is None:
        return None

    # Estimate tax rate
    tax = _get_ttm_sum(ctx.statements, "income_tax") or 0.0
    pretax = _get_ttm_sum(ctx.statements, "pretax_income") or 0.0
    tax_rate = max(0.0, min(0.35, tax / pretax)) if pretax > 0 else 0.21

    nopat = op_inc * (1.0 - tax_rate)

    debt = _get_latest_balance_val(ctx.statements, "total_debt") or 0.0
    equity = _get_latest_balance_val(ctx.statements, "total_equity") or 0.0
    cash = _get_latest_balance_val(ctx.statements, "cash") or 0.0

    invested_cap = debt + equity - cash
    if invested_cap <= 0:
        return None
    return float(nopat / invested_cap)


def calc_roe(ctx: MetricContext) -> Optional[float]:
    """Return on Equity (ROE). Higher is better (desc)."""
    ni = _get_ttm_sum(ctx.statements, "net_income")
    equity = _get_latest_balance_val(ctx.statements, "total_equity")
    if ni is None or equity is None or equity <= 0:
        return None
    return float(ni / equity)


def calc_operating_margin(ctx: MetricContext) -> Optional[float]:
    """Operating Margin. Higher is better (desc)."""
    op_inc = _get_ttm_sum(ctx.statements, "operating_income")
    rev = _get_ttm_sum(ctx.statements, "revenue")
    if op_inc is None or rev is None or rev <= 0:
        return None
    return float(op_inc / rev)


def calc_gross_margin(ctx: MetricContext) -> Optional[float]:
    """Gross Margin. Higher is better (desc)."""
    gp = _get_ttm_sum(ctx.statements, "gross_profit")
    rev = _get_ttm_sum(ctx.statements, "revenue")
    if gp is None or rev is None or rev <= 0:
        return None
    return float(gp / rev)


def calc_sales_growth_qoq(ctx: MetricContext) -> Optional[float]:
    """Quarter-over-Quarter Sales Growth. Higher is better (desc)."""
    q_stmts = [s for s in ctx.statements if s.period_type == "Q"]
    if len(q_stmts) < 2:
        return None
    r0 = q_stmts[0].data.get("revenue")
    r1 = q_stmts[1].data.get("revenue")
    if r0 is None or r1 is None or r1 <= 0:
        return None
    return float((r0 - r1) / r1)


def calc_sales_growth_yoy(ctx: MetricContext) -> Optional[float]:
    """Year-over-Year Sales Growth (Q vs Q-4). Higher is better (desc)."""
    q_stmts = [s for s in ctx.statements if s.period_type == "Q"]
    if len(q_stmts) < 5:
        # Fallback to annual statements if available
        fy = [s for s in ctx.statements if s.period_type == "FY"]
        if len(fy) >= 2:
            r0 = fy[0].data.get("revenue")
            r1 = fy[1].data.get("revenue")
            if r0 and r1 and r1 > 0:
                return float((r0 - r1) / r1)
        return None
    r0 = q_stmts[0].data.get("revenue")
    r4 = q_stmts[4].data.get("revenue")
    if r0 is None or r4 is None or r4 <= 0:
        return None
    return float((r0 - r4) / r4)


def calc_eps_growth_yoy(ctx: MetricContext) -> Optional[float]:
    """EPS Growth YoY (TTM vs TTM-1yr). Higher is better (desc)."""
    q_stmts = [s for s in ctx.statements if s.period_type == "Q"]
    if len(q_stmts) < 8:
        return None
    eps_current = sum(s.data.get("eps_diluted", 0) for s in q_stmts[:4])
    eps_prev = sum(s.data.get("eps_diluted", 0) for s in q_stmts[4:8])
    if eps_prev <= 0:
        return None
    return float((eps_current - eps_prev) / eps_prev)


def calc_debt_to_equity(ctx: MetricContext) -> Optional[float]:
    """Debt to Equity Ratio. Lower is better (asc)."""
    debt = _get_latest_balance_val(ctx.statements, "total_debt")
    equity = _get_latest_balance_val(ctx.statements, "total_equity")
    if debt is None or equity is None or equity <= 0:
        return None
    return float(debt / equity)


def calc_fcf_yield(ctx: MetricContext) -> Optional[float]:
    """Free Cash Flow Yield = FCF TTM / Market Cap. Higher is better (desc)."""
    if not ctx.price or ctx.price <= 0:
        return None
    ocf = _get_ttm_sum(ctx.statements, "operating_cash_flow")
    capex = _get_ttm_sum(ctx.statements, "capex") or 0.0
    shares = _get_latest_balance_val(ctx.statements, "shares_diluted")
    if ocf is None or shares is None or shares <= 0:
        return None
    fcf = ocf - capex
    mkt_cap = ctx.price * shares
    if mkt_cap <= 0:
        return None
    return float(fcf / mkt_cap)


# ---------------------------------------------------------------------------
# Price-derived metrics
# ---------------------------------------------------------------------------
def calc_dist_52w_low(ctx: MetricContext) -> Optional[float]:
    """Price distance from 52-week low = (Price / 52w_Low) - 1. Lower is better (asc)."""
    if not ctx.price or ctx.bars_df is None or ctx.bars_df.empty:
        return None
    # 252 trading days ~ 52 weeks
    tail_bars = ctx.bars_df.tail(252)
    low_52w = tail_bars["Low"].min() if "Low" in tail_bars.columns else tail_bars["Close"].min()
    if low_52w <= 0:
        return None
    return float((ctx.price / low_52w) - 1.0)


def calc_dist_52w_high(ctx: MetricContext) -> Optional[float]:
    """Price distance from 52-week high = (Price / 52w_High) - 1. Higher is better (desc) (closer to breakout)."""
    if not ctx.price or ctx.bars_df is None or ctx.bars_df.empty:
        return None
    tail_bars = ctx.bars_df.tail(252)
    high_52w = tail_bars["High"].max() if "High" in tail_bars.columns else tail_bars["Close"].max()
    if high_52w <= 0:
        return None
    return float((ctx.price / high_52w) - 1.0)


def calc_momentum_6m(ctx: MetricContext) -> Optional[float]:
    """6-Month Price Momentum (~126 trading days). Higher is better (desc)."""
    if ctx.bars_df is None or len(ctx.bars_df) < 126:
        return None
    c_now = ctx.bars_df["Close"].iloc[-1]
    c_past = ctx.bars_df["Close"].iloc[-126]
    if c_past <= 0:
        return None
    return float((c_now - c_past) / c_past)


def calc_momentum_12m(ctx: MetricContext) -> Optional[float]:
    """12-Month Price Momentum (~252 trading days). Higher is better (desc)."""
    if ctx.bars_df is None or len(ctx.bars_df) < 252:
        return None
    c_now = ctx.bars_df["Close"].iloc[-1]
    c_past = ctx.bars_df["Close"].iloc[-252]
    if c_past <= 0:
        return None
    return float((c_now - c_past) / c_past)


def calc_volatility_3m(ctx: MetricContext) -> Optional[float]:
    """3-Month Annualized Volatility (~63 trading days). Lower is better (asc)."""
    if ctx.bars_df is None or len(ctx.bars_df) < 63:
        return None
    returns = ctx.bars_df["Close"].tail(63).pct_change().dropna()
    if returns.empty:
        return None
    daily_std = float(returns.std())
    return float(daily_std * np.sqrt(252))


# ---------------------------------------------------------------------------
# Central Registry
# ---------------------------------------------------------------------------
METRIC_REGISTRY: Dict[str, MetricDef] = {
    "per": MetricDef(
        key="per",
        label="PER (Price-to-Earnings)",
        direction="asc",
        source="mixed",
        description="Ratio precio/beneficio TTM. Menor es mejor.",
        fn=calc_per,
    ),
    "peg": MetricDef(
        key="peg",
        label="PEG Ratio",
        direction="asc",
        source="mixed",
        description="PER ajustado por crecimiento anual de EPS. Menor es mejor.",
        fn=calc_peg,
    ),
    "roic": MetricDef(
        key="roic",
        label="ROIC (Return on Invested Capital)",
        direction="desc",
        source="fundamental",
        description="Retorno sobre el capital invertido TTM. Mayor es mejor.",
        fn=calc_roic,
    ),
    "roe": MetricDef(
        key="roe",
        label="ROE (Return on Equity)",
        direction="desc",
        source="fundamental",
        description="Retorno sobre el patrimonio neto TTM. Mayor es mejor.",
        fn=calc_roe,
    ),
    "operating_margin": MetricDef(
        key="operating_margin",
        label="Margen Operativo",
        direction="desc",
        source="fundamental",
        description="Beneficio operativo sobre ingresos TTM. Mayor es mejor.",
        fn=calc_operating_margin,
    ),
    "gross_margin": MetricDef(
        key="gross_margin",
        label="Margen Bruto",
        direction="desc",
        source="fundamental",
        description="Margen bruto sobre ventas TTM. Mayor es mejor.",
        fn=calc_gross_margin,
    ),
    "sales_growth_qoq": MetricDef(
        key="sales_growth_qoq",
        label="Crecimiento Ventas QoQ",
        direction="desc",
        source="fundamental",
        description="Crecimiento de ingresos respecto al trimestre anterior. Mayor es mejor.",
        fn=calc_sales_growth_qoq,
    ),
    "sales_growth_yoy": MetricDef(
        key="sales_growth_yoy",
        label="Crecimiento Ventas YoY",
        direction="desc",
        source="fundamental",
        description="Crecimiento de ingresos respecto al mismo trimestre del año anterior. Mayor es mejor.",
        fn=calc_sales_growth_yoy,
    ),
    "eps_growth_yoy": MetricDef(
        key="eps_growth_yoy",
        label="Crecimiento EPS YoY",
        direction="desc",
        source="fundamental",
        description="Crecimiento de beneficios por acción TTM anual. Mayor es mejor.",
        fn=calc_eps_growth_yoy,
    ),
    "debt_to_equity": MetricDef(
        key="debt_to_equity",
        label="Deuda / Patrimonio (D/E)",
        direction="asc",
        source="fundamental",
        description="Apalancamiento financiero. Menor es mejor.",
        fn=calc_debt_to_equity,
    ),
    "fcf_yield": MetricDef(
        key="fcf_yield",
        label="Free Cash Flow Yield",
        direction="desc",
        source="mixed",
        description="Flujo de caja libre sobre capitalización bursátil. Mayor es mejor.",
        fn=calc_fcf_yield,
    ),
    "dist_52w_low": MetricDef(
        key="dist_52w_low",
        label="Distancia a Mínimo 52 Semanas",
        direction="asc",
        source="price",
        description="Distancia porcentual al mínimo anual. Menor es mejor (cerca de soporte).",
        fn=calc_dist_52w_low,
    ),
    "dist_52w_high": MetricDef(
        key="dist_52w_high",
        label="Distancia a Máximo 52 Semanas",
        direction="desc",
        source="price",
        description="Proximidad al máximo de 52 semanas. Mayor es mejor (cerca de máximos).",
        fn=calc_dist_52w_high,
    ),
    "momentum_6m": MetricDef(
        key="momentum_6m",
        label="Momentum 6 Meses",
        direction="desc",
        source="price",
        description="Rendimiento total del precio en los últimos 6 meses. Mayor es mejor.",
        fn=calc_momentum_6m,
    ),
    "momentum_12m": MetricDef(
        key="momentum_12m",
        label="Momentum 12 Meses",
        direction="desc",
        source="price",
        description="Rendimiento total del precio en los últimos 12 meses. Mayor es mejor.",
        fn=calc_momentum_12m,
    ),
    "volatility_3m": MetricDef(
        key="volatility_3m",
        label="Volatilidad Anualizada 3M",
        direction="asc",
        source="price",
        description="Desviación típica anualizada a 63 días. Menor es mejor.",
        fn=calc_volatility_3m,
    ),
}


def get_metrics_catalog() -> List[Dict[str, Any]]:
    """Returns the catalog metadata format expected by the frontend / API."""
    return [
        {
            "key": m.key,
            "label": m.label,
            "direction": m.direction,
            "source": m.source,
            "description": m.description,
        }
        for m in METRIC_REGISTRY.values()
    ]
