from django.db import models
from core.models import TimeStampedModel, ImmutableModel
from apps.portfolios.models import Portfolio
from apps.brokers.models import BrokerAccount
from apps.strategies.models import Strategy
from apps.instruments.models import Instrument


class RebalanceRun(TimeStampedModel):
    TRIGGER_CHOICES = (
        ("scheduled", "Scheduled Daily Window"),
        ("allocation_change", "Allocation Weight Change"),
        ("signal_change", "Signal Change"),
        ("periodic", "Periodic Calendar Rebalance"),
        ("manual", "Manual User / Operator Trigger"),
    )
    STATUS_CHOICES = (
        ("planned", "Planned"),
        ("executing", "Executing"),
        ("done", "Done"),
        ("failed", "Failed"),
        ("missed_cutoff", "Missed Cutoff"),
    )

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="rebalance_runs")
    trigger = models.CharField(max_length=32, choices=TRIGGER_CHOICES, default="scheduled")
    as_of = models.DateTimeField(db_index=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="planned")
    equity_base = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    prices = models.JSONField(default=dict, blank=True)
    decisions = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Run #{self.id} {self.portfolio.name} [{self.trigger}] @ {self.as_of}"


class Target(ImmutableModel):
    """Calculated position target per sleeve (strategy) and instrument."""
    run = models.ForeignKey(RebalanceRun, on_delete=models.CASCADE, related_name="targets")
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    broker_account = models.ForeignKey(BrokerAccount, on_delete=models.SET_NULL, null=True, blank=True)
    sleeve_capital = models.DecimalField(max_digits=16, decimal_places=2)
    target_qty = models.DecimalField(max_digits=16, decimal_places=8)
    current_qty = models.DecimalField(max_digits=16, decimal_places=8)
    delta_qty = models.DecimalField(max_digits=16, decimal_places=8)
    reason = models.CharField(max_length=128, default="drift")

    def __str__(self):
        return f"Target {self.strategy.slug} -> {self.instrument.symbol}: delta={self.delta_qty} ({self.reason})"


class OrderIntent(models.Model):
    """Netted net delta per (broker_account, instrument) with deterministic idempotency key."""
    STATUS_CHOICES = (
        ("pending", "Pending"),
        ("dispatched", "Dispatched"),
        ("cancelled", "Cancelled"),
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    run = models.ForeignKey(RebalanceRun, on_delete=models.CASCADE, related_name="intents")
    broker_account = models.ForeignKey(BrokerAccount, on_delete=models.CASCADE, null=True, blank=True)
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    net_delta_qty = models.DecimalField(max_digits=16, decimal_places=8)
    idempotency_key = models.CharField(max_length=128, unique=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="pending")

    def __str__(self):
        return f"Intent {self.instrument.symbol} net={self.net_delta_qty} [{self.idempotency_key}]"


class Order(TimeStampedModel):
    SOURCE_CHOICES = (
        ("broker", "Real Broker API (Alpaca MOC)"),
        ("simulated_manual", "Simulated Manual (LatestQuote + Telegram)"),
        ("simulated_paper", "Simulated Paper App (Official Close)"),
    )
    SIDE_CHOICES = (
        ("buy", "Buy"),
        ("sell", "Sell"),
    )
    STATUS_CHOICES = (
        ("submitted", "Submitted"),
        ("accepted", "Accepted"),
        ("filled", "Filled"),
        ("cancelled", "Cancelled"),
        ("rejected", "Rejected"),
    )

    intent = models.OneToOneField(OrderIntent, on_delete=models.SET_NULL, null=True, blank=True, related_name="order")
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="orders")
    broker_account = models.ForeignKey(BrokerAccount, on_delete=models.SET_NULL, null=True, blank=True)
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    client_order_id = models.CharField(max_length=128, unique=True)
    broker_order_id = models.CharField(max_length=128, blank=True)
    source = models.CharField(max_length=32, choices=SOURCE_CHOICES, default="broker")
    order_type = models.CharField(max_length=16, default="moc")
    side = models.CharField(max_length=8, choices=SIDE_CHOICES)
    qty = models.DecimalField(max_digits=16, decimal_places=8)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="submitted")

    def __str__(self):
        return f"Order {self.client_order_id} {self.side.upper()} {self.qty} {self.instrument.symbol} [{self.status}]"


class Fill(ImmutableModel):
    """Immutable execution event record (real or simulated)."""
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="fills")
    fill_price = models.DecimalField(max_digits=16, decimal_places=6)
    qty = models.DecimalField(max_digits=16, decimal_places=8)
    commission = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    source = models.CharField(max_length=32, default="broker")
    filled_at = models.DateTimeField(db_index=True)

    def __str__(self):
        return f"Fill #{self.id} for Order {self.order.client_order_id}: {self.qty} @ ${self.fill_price}"


class FillAllocation(ImmutableModel):
    """Attribution of executed fill quantities back to individual strategy sleeves."""
    fill = models.ForeignKey(Fill, on_delete=models.CASCADE, related_name="allocations")
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    allocated_qty = models.DecimalField(max_digits=16, decimal_places=8)

    def __str__(self):
        return f"FillAlloc {self.strategy.slug}: {self.allocated_qty} shares"
