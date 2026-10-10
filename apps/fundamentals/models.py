from django.db import models
from core.models import TimeStampedModel
from apps.instruments.models import Instrument


# Canonical base fields stored for every statement, independent of provider.
# Derived metrics (PER, ROIC, margins...) are computed locally in metrics.py.
CANONICAL_FIELDS = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "pretax_income",
    "income_tax",
    "interest_expense",
    "ebitda",
    "eps_diluted",
    "shares_diluted",
    "total_assets",
    "total_equity",
    "total_debt",
    "cash",
    "current_assets",
    "current_liabilities",
    "operating_cash_flow",
    "capex",
    "dividends_paid",
)


class FundamentalStatement(TimeStampedModel):
    """One fiscal period of normalized base fundamentals for an instrument."""
    PERIOD_CHOICES = (("Q", "Quarter"), ("FY", "Fiscal Year"))
    SOURCE_CHOICES = (("fmp", "FMP"), ("eodhd", "EODHD"), ("yfinance", "yfinance"))

    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, related_name="fundamentals")
    period_type = models.CharField(max_length=2, choices=PERIOD_CHOICES, default="Q")
    period_end = models.DateField(db_index=True)
    available_at = models.DateField(null=True, blank=True, db_index=True,
                                    help_text="Fecha de publicación (filing). Si falta se estima period_end + 45 días.")
    source = models.CharField(max_length=16, choices=SOURCE_CHOICES)
    data = models.JSONField(default=dict)
    raw = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = ("instrument", "period_type", "period_end")
        ordering = ["-period_end"]

    def __str__(self):
        return f"{self.instrument.symbol} {self.period_type} {self.period_end} [{self.source}]"


class FundamentalFetchLog(TimeStampedModel):
    """Audit + quota tracking of provider calls."""
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, null=True, blank=True)
    source = models.CharField(max_length=16)
    status = models.CharField(max_length=16, default="ok")  # ok | error | quota | empty
    calls_used = models.PositiveIntegerField(default=0)
    statements_saved = models.PositiveIntegerField(default=0)
    error = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        sym = self.instrument.symbol if self.instrument else "-"
        return f"{self.source} {sym} [{self.status}] calls={self.calls_used}"
