from django.db import models
from core.models import TimeStampedModel, ImmutableModel
from apps.portfolios.models import Portfolio
from apps.brokers.models import BrokerAccount
from apps.strategies.models import Strategy


class KillSwitch(TimeStampedModel):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, null=True, blank=True)
    broker_account = models.ForeignKey(BrokerAccount, on_delete=models.CASCADE, null=True, blank=True)
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, null=True, blank=True)
    reason = models.TextField()
    is_active = models.BooleanField(default=True)
    activated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        target = self.portfolio or self.broker_account or self.strategy or "SYSTEM GLOBAL"
        return f"KillSwitch [{target}]: {'ACTIVE' if self.is_active else 'RESOLVED'} ({self.reason[:30]})"


class ReconciliationReport(ImmutableModel):
    broker_account = models.ForeignKey(BrokerAccount, on_delete=models.CASCADE, related_name="reconciliation_reports")
    date = models.DateField(db_index=True)
    matched = models.BooleanField(default=True)
    discrepancy_details = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Reconciliation {self.broker_account.display_name} @ {self.date}: {'MATCH' if self.matched else 'MISMATCH'}"


class FundingAlert(TimeStampedModel):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
    broker_account = models.ForeignKey(BrokerAccount, on_delete=models.CASCADE, null=True, blank=True)
    required_cash = models.DecimalField(max_digits=16, decimal_places=2)
    available_cash = models.DecimalField(max_digits=16, decimal_places=2)
    status = models.CharField(max_length=16, default="open")

    def __str__(self):
        return f"FundingAlert #{self.id} {self.portfolio.name}: Need ${self.required_cash}, Have ${self.available_cash}"
