from django.db import models
from django.db.models import Q
from django.conf import settings
from django.core.exceptions import ValidationError
from core.models import TimeStampedModel, ImmutableModel
from apps.strategies.models import Strategy, StrategyVersion


class RecommendedPortfolio(TimeStampedModel):
    REBALANCE_FREQ_CHOICES = (
        ("daily", "Daily"),
        ("weekly", "Weekly (Last business day)"),
        ("monthly", "Monthly (Last business day)"),
        ("quarterly", "Quarterly (Last business day)"),
        ("never", "Never / Signal Trigger Only"),
    )
    TOLERANCE_MODE_CHOICES = (
        ("relative", "Relative Drift"),
        ("absolute", "Absolute Drift"),
    )

    slug = models.SlugField(unique=True, max_length=64)
    name = models.CharField(max_length=128)
    description = models.TextField(blank=True)
    risk_level = models.CharField(max_length=32, default="moderate")
    rebalance_frequency = models.CharField(max_length=32, choices=REBALANCE_FREQ_CHOICES, default="monthly")
    rebalance_tolerance = models.DecimalField(max_digits=7, decimal_places=4, default=0.1000)
    rebalance_tolerance_mode = models.CharField(max_length=16, choices=TOLERANCE_MODE_CHOICES, default="relative")
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} ({self.slug})"


class RecommendedAllocation(TimeStampedModel):
    recommended = models.ForeignKey(RecommendedPortfolio, on_delete=models.CASCADE, related_name="allocations")
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    target_weight = models.DecimalField(max_digits=7, decimal_places=4)

    class Meta:
        unique_together = ("recommended", "strategy")

    def __str__(self):
        return f"{self.recommended.name} - {self.strategy.name}: {self.target_weight * 100}%"


class Portfolio(TimeStampedModel):
    KIND_CHOICES = (
        ("live", "Live (Real Money)"),
        ("paper", "Paper (Simulated)"),
        ("sandbox", "Sandbox (Staff Broker Paper)"),
    )
    STATUS_CHOICES = (
        ("active", "Active"),
        ("suspended", "Suspended"),
        ("closed", "Closed"),
    )
    REBALANCE_FREQ_CHOICES = (
        ("daily", "Daily"),
        ("weekly", "Weekly (Last business day)"),
        ("monthly", "Monthly (Last business day)"),
        ("quarterly", "Quarterly (Last business day)"),
        ("never", "Never / Signal Trigger Only"),
    )
    TOLERANCE_MODE_CHOICES = (
        ("relative", "Relative Drift"),
        ("absolute", "Absolute Drift"),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="portfolios")
    kind = models.CharField(max_length=16, choices=KIND_CHOICES, default="paper")
    name = models.CharField(max_length=128)
    base_currency = models.CharField(max_length=8, default="USD")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="active")
    trading_enabled = models.BooleanField(default=True)
    rebalance_frequency = models.CharField(max_length=32, choices=REBALANCE_FREQ_CHOICES, default="daily")
    rebalance_tolerance = models.DecimalField(max_digits=7, decimal_places=4, default=0.1000)
    rebalance_tolerance_mode = models.CharField(max_length=16, choices=TOLERANCE_MODE_CHOICES, default="relative")
    max_gross_allocation = models.DecimalField(max_digits=7, decimal_places=4, default=1.0000)
    paper_initial_capital = models.DecimalField(max_digits=16, decimal_places=2, default=100000.00)
    paper_commission_model = models.JSONField(default=dict, blank=True)
    follows = models.ForeignKey(RecommendedPortfolio, on_delete=models.SET_NULL, null=True, blank=True)
    rebalance_pending = models.BooleanField(default=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user"],
                condition=Q(kind="live") & ~Q(status="closed"),
                name="one_live_portfolio_per_user",
            )
        ]

    def clean(self):
        super().clean()
        if self.pk is None and self.user_id:
            user = self.user
            plan = user.effective_plan
            if self.kind == "live":
                active_lives = Portfolio.objects.filter(user=user, kind="live").exclude(status="closed").count()
                if active_lives >= plan.max_live_portfolios:
                    raise ValidationError(f"Plan limit exceeded: Maximum {plan.max_live_portfolios} active Live portfolio allowed.")
            elif self.kind == "paper":
                active_papers = Portfolio.objects.filter(user=user, kind="paper").exclude(status="closed").count()
                if active_papers >= plan.max_paper_portfolios:
                    raise ValidationError(f"Plan limit exceeded: Maximum {plan.max_paper_portfolios} Paper portfolios allowed.")

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)

    def calculate_drift_metrics(self):
        """
        Calculates current vs target allocation drift across all sleeves in this portfolio.
        Returns a dict with total_equity, allocations breakdown, and overall rebalance_needed boolean.
        """
        from decimal import Decimal
        from apps.accounting.models import LogicalPosition
        from apps.marketdata.models import LatestQuote

        # Base equity estimate
        total_equity = self.paper_initial_capital if self.kind == "paper" else Decimal("100000.00")
        
        allocations = self.allocations.filter(enabled=True).select_related("strategy")
        breakdown = []
        any_breached = False

        total_positions_val = Decimal("0.0")
        # First calculate current values of all sleeves
        sleeve_values = {}
        for alloc in allocations:
            positions = LogicalPosition.objects.filter(portfolio=self, strategy=alloc.strategy)
            val = Decimal("0.0")
            for pos in positions:
                lq = LatestQuote.objects.filter(instrument=pos.instrument).first()
                price = lq.price if lq else Decimal("100.00")
                val += pos.qty * price
            sleeve_values[alloc.id] = val
            total_positions_val += val

        effective_equity = max(total_equity, total_positions_val)

        for alloc in allocations:
            current_val = sleeve_values.get(alloc.id, Decimal("0.0"))
            current_weight = current_val / effective_equity if effective_equity > Decimal("0") else Decimal("0")
            target_weight = alloc.target_weight
            target_val = effective_equity * target_weight

            abs_drift = abs(current_weight - target_weight)
            rel_drift = abs_drift / target_weight if target_weight > Decimal("0.0001") else (Decimal("1.0") if current_val > 0 else Decimal("0"))
            
            if self.rebalance_tolerance_mode == "relative":
                breached = rel_drift > self.rebalance_tolerance
            else:
                breached = abs_drift > self.rebalance_tolerance

            if breached:
                any_breached = True

            breakdown.append({
                "allocation_id": alloc.id,
                "strategy_id": alloc.strategy_id,
                "strategy_name": alloc.strategy.name,
                "strategy_slug": alloc.strategy.slug,
                "target_weight": float(target_weight),
                "target_value": float(target_val),
                "current_value": float(current_val),
                "current_weight": float(round(current_weight, 4)),
                "absolute_drift": float(round(abs_drift, 4)),
                "relative_drift": float(round(rel_drift, 4)),
                "tolerance": float(self.rebalance_tolerance),
                "tolerance_mode": self.rebalance_tolerance_mode,
                "breached": breached,
            })

        return {
            "portfolio_id": self.id,
            "portfolio_name": self.name,
            "rebalance_frequency": self.rebalance_frequency,
            "rebalance_tolerance": float(self.rebalance_tolerance),
            "rebalance_tolerance_mode": self.rebalance_tolerance_mode,
            "total_equity": float(effective_equity),
            "rebalance_needed": any_breached,
            "allocations": breakdown,
        }

    def __str__(self):
        return f"{self.user.email} - {self.name} [{self.kind.upper()}]"


class Allocation(TimeStampedModel):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="allocations")
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    version = models.ForeignKey(StrategyVersion, on_delete=models.SET_NULL, null=True, blank=True)
    target_weight = models.DecimalField(max_digits=7, decimal_places=4)
    broker_account = models.ForeignKey("brokers.BrokerAccount", on_delete=models.SET_NULL, null=True, blank=True)
    enabled = models.BooleanField(default=True)

    class Meta:
        unique_together = ("portfolio", "strategy")

    def __str__(self):
        return f"{self.portfolio.name} -> {self.strategy.slug}: {self.target_weight * 100}%"


class AllocationChange(ImmutableModel):
    SOURCE_CHOICES = (
        ("user", "User Request"),
        ("recommended_follow", "Follow Recommended Portfolio"),
        ("admin", "Admin Override"),
    )
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="allocation_changes")
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    old_weight = models.DecimalField(max_digits=7, decimal_places=4)
    new_weight = models.DecimalField(max_digits=7, decimal_places=4)
    source = models.CharField(max_length=32, choices=SOURCE_CHOICES, default="user")
    effective_at = models.DateTimeField()
    applied = models.BooleanField(default=False)

    def __str__(self):
        return f"Change {self.portfolio.name} {self.strategy.slug}: {self.old_weight} -> {self.new_weight} at {self.effective_at}"
