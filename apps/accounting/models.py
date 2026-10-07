from django.db import models
from core.models import TimeStampedModel, ImmutableModel
from apps.portfolios.models import Portfolio
from apps.brokers.models import BrokerAccount
from apps.strategies.models import Strategy
from apps.instruments.models import Instrument


class LogicalPosition(TimeStampedModel):
    """
    Virtual bookkeeping of position quantities per (portfolio, broker_account, strategy, instrument).
    """
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="logical_positions")
    broker_account = models.ForeignKey(BrokerAccount, on_delete=models.CASCADE, null=True, blank=True)
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    qty = models.DecimalField(max_digits=16, decimal_places=8, default=0)
    avg_cost = models.DecimalField(max_digits=16, decimal_places=6, default=0)
    realized_pnl = models.DecimalField(max_digits=16, decimal_places=2, default=0)

    class Meta:
        unique_together = ("portfolio", "broker_account", "strategy", "instrument")

    def __str__(self):
        return f"Pos {self.portfolio.name} | {self.strategy.slug} | {self.instrument.symbol}: {self.qty}"


class BrokerPositionSnapshot(ImmutableModel):
    """Broker reported position snapshot used for daily reconciliations."""
    broker_account = models.ForeignKey(BrokerAccount, on_delete=models.CASCADE, related_name="position_snapshots")
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    qty = models.DecimalField(max_digits=16, decimal_places=8)
    snapshot_time = models.DateTimeField(db_index=True)

    def __str__(self):
        return f"Snapshot {self.broker_account.display_name} {self.instrument.symbol}: {self.qty}"


class LedgerEntry(ImmutableModel):
    """Double-entry accounting ledger entries."""
    ENTRY_TYPE_CHOICES = (
        ("deposit", "Deposit"),
        ("withdrawal", "Withdrawal"),
        ("realized_pnl", "Realized PnL"),
        ("fee", "Fee / Commission"),
        ("opening_balance", "Opening Balance"),
    )
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="ledger_entries")
    broker_account = models.ForeignKey(BrokerAccount, on_delete=models.CASCADE, null=True, blank=True)
    entry_type = models.CharField(max_length=32, choices=ENTRY_TYPE_CHOICES)
    amount = models.DecimalField(max_digits=16, decimal_places=2)
    currency = models.CharField(max_length=8, default="USD")
    reference = models.CharField(max_length=128, blank=True)

    def __str__(self):
        return f"Ledger #{self.id} {self.portfolio.name} {self.entry_type}: ${self.amount}"


class EquitySnapshot(TimeStampedModel):
    """Daily total portfolio equity curve snapshot."""
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="equity_snapshots")
    date = models.DateField(db_index=True)
    total_equity = models.DecimalField(max_digits=16, decimal_places=2)
    cash_balance = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    positions_value = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    source = models.CharField(max_length=32, default="daily_cycle")

    class Meta:
        unique_together = ("portfolio", "date")
        ordering = ["-date"]

    def __str__(self):
        return f"Equity {self.portfolio.name} @ {self.date}: ${self.total_equity}"


class SleevePnlDaily(TimeStampedModel):
    """Daily PnL per strategy sleeve within a portfolio."""
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="sleeve_pnls")
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    date = models.DateField(db_index=True)
    pnl = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    sleeve_value = models.DecimalField(max_digits=16, decimal_places=2, default=0)

    class Meta:
        unique_together = ("portfolio", "strategy", "date")

    def __str__(self):
        return f"Sleeve {self.portfolio.name} - {self.strategy.slug} @ {self.date}: ${self.sleeve_value} (pnl: ${self.pnl})"


class BenchmarkDaily(TimeStampedModel):
    """Daily benchmark prices and cumulative returns for SPY and QQQ."""
    symbol = models.CharField(max_length=16, db_index=True)
    date = models.DateField(db_index=True)
    close = models.DecimalField(max_digits=16, decimal_places=6)
    cumulative_return = models.DecimalField(max_digits=10, decimal_places=4, default=0)

    class Meta:
        unique_together = ("symbol", "date")
        ordering = ["-date"]

    def __str__(self):
        return f"Benchmark {self.symbol} @ {self.date}: {self.close}"
