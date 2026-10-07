from django.db import models
from core.models import TimeStampedModel, ImmutableModel
from apps.instruments.models import Instrument


class Strategy(TimeStampedModel):
    KIND_CHOICES = (
        ("one_strategy", "One Strategy Bot"),
        ("multi_strategy", "Multi Strategy Bot"),
        ("cross_asset", "Cross Asset Bot"),
        ("follow_price", "Follow Price Bot"),
        ("signal_dollar", "Signal Dollar (Macro)"),
        ("signal_options", "Signal Options (Macro)"),
    )

    slug = models.SlugField(unique=True, max_length=64)
    name = models.CharField(max_length=128)
    family = models.CharField(max_length=64, db_index=True, default="General")
    kind = models.CharField(max_length=32, choices=KIND_CHOICES, default="one_strategy")
    engine = models.CharField(max_length=64, default="single_signal")
    timeframe = models.CharField(max_length=16, default="1d")
    decision_offset_minutes = models.PositiveIntegerField(default=25)
    is_active = models.BooleanField(default=True)
    signal_only = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.name} ({self.slug}) [{self.get_kind_display()}]"


class StrategyVersion(TimeStampedModel):
    STATUS_CHOICES = (
        ("draft", "Draft"),
        ("paper", "Paper"),
        ("live", "Live"),
        ("retired", "Retired"),
    )
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, related_name="versions")
    version = models.PositiveIntegerField(default=1)
    params = models.JSONField(default=dict, blank=True)
    params_schema = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="live")

    class Meta:
        unique_together = ("strategy", "version")
        ordering = ["-version"]

    def __str__(self):
        return f"{self.strategy.slug} v{self.version} [{self.status}]"


class StrategyInstrument(TimeStampedModel):
    ROLE_CHOICES = (
        ("traded", "Traded Asset"),
        ("signal_source", "Signal Source Asset"),
    )
    version = models.ForeignKey(StrategyVersion, on_delete=models.CASCADE, related_name="instruments")
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    role = models.CharField(max_length=16, choices=ROLE_CHOICES, default="traded")
    weight = models.DecimalField(max_digits=7, decimal_places=4, default=1.0000)
    params = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = ("version", "instrument", "role")

    def __str__(self):
        return f"{self.version} - {self.instrument.symbol} ({self.role}, weight: {self.weight})"


class Signal(ImmutableModel):
    """Immutable audit record of a strategy signal calculated on a specific date."""
    version = models.ForeignKey(StrategyVersion, on_delete=models.CASCADE, related_name="signals")
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE)
    as_of = models.DateTimeField(db_index=True)
    target_exposure = models.DecimalField(max_digits=7, decimal_places=4)
    changed = models.BooleanField(default=False)
    diagnostics = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-as_of"]

    def __str__(self):
        return f"Signal {self.version.strategy.slug} -> {self.instrument.symbol}: {self.target_exposure} @ {self.as_of}"


class BacktestRun(TimeStampedModel):
    STATUS_CHOICES = (
        ("queued", "Queued"),
        ("running", "Running"),
        ("done", "Done"),
        ("failed", "Failed"),
    )
    version = models.ForeignKey(StrategyVersion, on_delete=models.SET_NULL, null=True, blank=True)
    strategy_slug = models.CharField(max_length=64)
    instruments = models.JSONField(default=list)
    start_date = models.DateField()
    end_date = models.DateField()
    initial_capital = models.DecimalField(max_digits=16, decimal_places=2, default=100000.00)
    rebalance_override = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="queued")
    metrics = models.JSONField(default=dict, blank=True)
    equity_curve = models.JSONField(default=list, blank=True)
    error_message = models.TextField(blank=True)

    def __str__(self):
        return f"Backtest {self.strategy_slug} ({self.start_date} to {self.end_date}) [{self.status}]"
