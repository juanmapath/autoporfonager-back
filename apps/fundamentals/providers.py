"""
Fundamental data providers.

Every provider returns a list of CanonicalStatement with fields from
`apps.fundamentals.models.CANONICAL_FIELDS`, so derived metrics are computed
identically regardless of the source. Chain order: FMP -> EODHD -> yfinance.
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 20
# Minimum fields required to consider a statement usable.
CRITICAL_FIELDS = ("revenue", "net_income", "total_equity")


class ProviderError(Exception):
    pass


class QuotaExceeded(ProviderError):
    pass


@dataclass
class CanonicalStatement:
    period_type: str  # "Q" | "FY"
    period_end: date
    available_at: Optional[date]
    data: Dict[str, Optional[float]]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FetchResult:
    source: str
    statements: List[CanonicalStatement]
    calls_used: int


def _num(v) -> Optional[float]:
    if v is None or v == "" or v == "None":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _date(v) -> Optional[date]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _first(d: Dict[str, Any], *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def is_usable(st: CanonicalStatement) -> bool:
    return all(st.data.get(f) is not None for f in CRITICAL_FIELDS)


class BaseProvider:
    name = "base"

    def available(self) -> bool:
        return True

    def fetch(self, symbol: str) -> FetchResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FMP (Financial Modeling Prep) — stable API
# ---------------------------------------------------------------------------
class FMPProvider(BaseProvider):
    name = "fmp"
    BASE = "https://financialmodelingprep.com/stable"

    def available(self) -> bool:
        return bool(getattr(settings, "FMP_API_KEY", ""))

    def _get(self, endpoint: str, symbol: str, period: str) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "period": period, "limit": 12, "apikey": settings.FMP_API_KEY}
        r = requests.get(f"{self.BASE}/{endpoint}", params=params, timeout=HTTP_TIMEOUT)
        if r.status_code == 429:
            raise QuotaExceeded("FMP rate limit / daily quota exceeded")
        if r.status_code in (401, 402, 403):
            raise ProviderError(f"FMP access denied ({r.status_code}) for {endpoint} period={period}: {r.text[:200]}")
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and ("Error Message" in payload or "error" in payload):
            msg = str(payload)[:200]
            if "limit" in msg.lower():
                raise QuotaExceeded(msg)
            raise ProviderError(msg)
        return payload if isinstance(payload, list) else []

    def fetch(self, symbol: str) -> FetchResult:
        calls = 0
        last_err: Optional[Exception] = None
        # Free tier may not include quarterly statements -> fall back to annual.
        for period, ptype in (("quarter", "Q"), ("annual", "FY")):
            try:
                inc = self._get("income-statement", symbol, period); calls += 1
                bal = self._get("balance-sheet-statement", symbol, period); calls += 1
                cfs = self._get("cash-flow-statement", symbol, period); calls += 1
            except QuotaExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
            if not inc:
                continue
            by_date_bal = {b.get("date"): b for b in bal}
            by_date_cf = {c.get("date"): c for c in cfs}
            out = []
            for i in inc:
                d = i.get("date")
                b = by_date_bal.get(d, {})
                c = by_date_cf.get(d, {})
                capex = _num(c.get("capitalExpenditure"))
                divs = _num(_first(c, "commonDividendsPaid", "dividendsPaid", "netDividendsPaid"))
                data = {
                    "revenue": _num(i.get("revenue")),
                    "gross_profit": _num(i.get("grossProfit")),
                    "operating_income": _num(i.get("operatingIncome")),
                    "net_income": _num(i.get("netIncome")),
                    "pretax_income": _num(i.get("incomeBeforeTax")),
                    "income_tax": _num(i.get("incomeTaxExpense")),
                    "interest_expense": _num(i.get("interestExpense")),
                    "ebitda": _num(i.get("ebitda")),
                    "eps_diluted": _num(_first(i, "epsDiluted", "epsdiluted", "eps")),
                    "shares_diluted": _num(_first(i, "weightedAverageShsOutDil", "weightedAverageShsOut")),
                    "total_assets": _num(b.get("totalAssets")),
                    "total_equity": _num(_first(b, "totalStockholdersEquity", "totalEquity")),
                    "total_debt": _num(b.get("totalDebt")),
                    "cash": _num(_first(b, "cashAndCashEquivalents", "cashAndShortTermInvestments")),
                    "current_assets": _num(b.get("totalCurrentAssets")),
                    "current_liabilities": _num(b.get("totalCurrentLiabilities")),
                    "operating_cash_flow": _num(c.get("operatingCashFlow")),
                    "capex": abs(capex) if capex is not None else None,
                    "dividends_paid": abs(divs) if divs is not None else None,
                }
                out.append(CanonicalStatement(
                    period_type=ptype,
                    period_end=_date(d),
                    available_at=_date(_first(i, "filingDate", "fillingDate", "acceptedDate")),
                    data=data,
                    raw={"income": i, "balance": b, "cashflow": c},
                ))
            out = [s for s in out if s.period_end]
            if out:
                return FetchResult(self.name, out, calls)
        raise ProviderError(f"FMP returned no statements for {symbol}: {last_err}")


# ---------------------------------------------------------------------------
# EODHD — single fundamentals call (costs 10 API credits)
# ---------------------------------------------------------------------------
class EODHDProvider(BaseProvider):
    name = "eodhd"
    BASE = "https://eodhd.com/api/fundamentals"

    def available(self) -> bool:
        return bool(getattr(settings, "EODHD_API_KEY", ""))

    def fetch(self, symbol: str) -> FetchResult:
        ticker = symbol if "." in symbol else f"{symbol}.US"
        r = requests.get(f"{self.BASE}/{ticker}",
                         params={"api_token": settings.EODHD_API_KEY, "fmt": "json"},
                         timeout=HTTP_TIMEOUT)
        if r.status_code in (402, 429):
            raise QuotaExceeded(f"EODHD quota ({r.status_code})")
        if r.status_code in (401, 403, 404):
            raise ProviderError(f"EODHD error {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        js = r.json() or {}
        fin = js.get("Financials") or {}
        inc_q = (fin.get("Income_Statement") or {}).get("quarterly") or {}
        bal_q = (fin.get("Balance_Sheet") or {}).get("quarterly") or {}
        cf_q = (fin.get("Cash_Flow") or {}).get("quarterly") or {}
        eps_hist = (js.get("Earnings") or {}).get("History") or {}

        out = []
        for d, i in sorted(inc_q.items(), reverse=True)[:12]:
            b = bal_q.get(d, {}) or {}
            c = cf_q.get(d, {}) or {}
            e = eps_hist.get(d, {}) or {}
            capex = _num(c.get("capitalExpenditures"))
            divs = _num(c.get("dividendsPaid"))
            data = {
                "revenue": _num(i.get("totalRevenue")),
                "gross_profit": _num(i.get("grossProfit")),
                "operating_income": _num(i.get("operatingIncome")),
                "net_income": _num(i.get("netIncome")),
                "pretax_income": _num(i.get("incomeBeforeTax")),
                "income_tax": _num(i.get("incomeTaxExpense")),
                "interest_expense": _num(i.get("interestExpense")),
                "ebitda": _num(i.get("ebitda")),
                "eps_diluted": _num(e.get("epsActual")),
                "shares_diluted": _num(b.get("commonStockSharesOutstanding")),
                "total_assets": _num(b.get("totalAssets")),
                "total_equity": _num(b.get("totalStockholderEquity")),
                "total_debt": _num(_first(b, "shortLongTermDebtTotal", "longTermDebt")),
                "cash": _num(_first(b, "cash", "cashAndEquivalents", "cashAndShortTermInvestments")),
                "current_assets": _num(b.get("totalCurrentAssets")),
                "current_liabilities": _num(b.get("totalCurrentLiabilities")),
                "operating_cash_flow": _num(c.get("totalCashFromOperatingActivities")),
                "capex": abs(capex) if capex is not None else None,
                "dividends_paid": abs(divs) if divs is not None else None,
            }
            out.append(CanonicalStatement(
                period_type="Q",
                period_end=_date(d),
                available_at=_date(i.get("filing_date")),
                data=data,
                raw={"income": i, "balance": b, "cashflow": c},
            ))
        out = [s for s in out if s.period_end]
        if not out:
            raise ProviderError(f"EODHD returned no statements for {symbol}")
        return FetchResult(self.name, out, 10)


# ---------------------------------------------------------------------------
# yfinance — last resort (no key, no filing dates)
# ---------------------------------------------------------------------------
class YFinanceProvider(BaseProvider):
    name = "yfinance"

    @staticmethod
    def _row(df, *names):
        if df is None or df.empty:
            return {}
        for n in names:
            if n in df.index:
                return df.loc[n].to_dict()
        return {}

    def fetch(self, symbol: str) -> FetchResult:
        import yfinance as yf

        t = yf.Ticker(symbol)
        inc = t.quarterly_income_stmt
        bal = t.quarterly_balance_sheet
        cf = t.quarterly_cashflow
        if inc is None or inc.empty:
            raise ProviderError(f"yfinance returned no statements for {symbol}")

        rows = {
            "revenue": self._row(inc, "Total Revenue", "Operating Revenue"),
            "gross_profit": self._row(inc, "Gross Profit"),
            "operating_income": self._row(inc, "Operating Income", "EBIT"),
            "net_income": self._row(inc, "Net Income", "Net Income Common Stockholders"),
            "pretax_income": self._row(inc, "Pretax Income"),
            "income_tax": self._row(inc, "Tax Provision"),
            "interest_expense": self._row(inc, "Interest Expense"),
            "ebitda": self._row(inc, "EBITDA", "Normalized EBITDA"),
            "eps_diluted": self._row(inc, "Diluted EPS"),
            "shares_diluted": self._row(inc, "Diluted Average Shares"),
            "total_assets": self._row(bal, "Total Assets"),
            "total_equity": self._row(bal, "Stockholders Equity", "Common Stock Equity"),
            "total_debt": self._row(bal, "Total Debt"),
            "cash": self._row(bal, "Cash And Cash Equivalents"),
            "current_assets": self._row(bal, "Current Assets"),
            "current_liabilities": self._row(bal, "Current Liabilities"),
            "operating_cash_flow": self._row(cf, "Operating Cash Flow"),
            "capex": self._row(cf, "Capital Expenditure"),
            "dividends_paid": self._row(cf, "Cash Dividends Paid", "Common Stock Dividend Paid"),
        }
        out = []
        for col in inc.columns:
            data = {}
            for k, row in rows.items():
                v = _num(row.get(col))
                if k in ("capex", "dividends_paid") and v is not None:
                    v = abs(v)
                data[k] = v
            pe = _date(col)
            out.append(CanonicalStatement(period_type="Q", period_end=pe, available_at=None, data=data))
        out = [s for s in out if s.period_end]
        return FetchResult(self.name, out, 1)


DEFAULT_CHAIN = (FMPProvider, EODHDProvider, YFinanceProvider)


class ProviderChain:
    """Tries providers in order; falls back on error, quota or unusable data."""

    def __init__(self, providers=None, skip: Optional[set] = None):
        self.providers = [p() for p in (providers or DEFAULT_CHAIN)]
        self.skip = skip or set()

    def fetch(self, symbol: str) -> (Optional[FetchResult], List[Dict[str, Any]]):
        attempts = []
        for p in self.providers:
            if p.name in self.skip or not p.available():
                continue
            try:
                res = p.fetch(symbol)
                usable = [s for s in res.statements if is_usable(s)]
                if not usable:
                    attempts.append({"source": p.name, "status": "empty", "calls": res.calls_used, "error": "no usable statements"})
                    continue
                res.statements = usable
                attempts.append({"source": p.name, "status": "ok", "calls": res.calls_used, "error": ""})
                return res, attempts
            except QuotaExceeded as e:
                attempts.append({"source": p.name, "status": "quota", "calls": 0, "error": str(e)})
                self.skip.add(p.name)  # do not retry this provider in the current sync run
            except Exception as e:  # noqa: BLE001
                logger.warning("Fundamental provider %s failed for %s: %s", p.name, symbol, e)
                attempts.append({"source": p.name, "status": "error", "calls": 0, "error": str(e)[:500]})
        return None, attempts


def estimate_available_at(period_end: date, period_type: str) -> date:
    """SEC deadlines approx: 10-Q ~40-45 days, 10-K ~60-90 days."""
    return period_end + timedelta(days=45 if period_type == "Q" else 75)
