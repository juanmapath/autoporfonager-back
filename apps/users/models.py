from django.db import models
from django.contrib.auth.models import AbstractUser
from core.models import TimeStampedModel


class Plan(TimeStampedModel):
    PLAN_CHOICES = (
        ("default", "Default / Free"),
        ("pro", "Pro"),
        ("internal", "Internal / Staff"),
    )
    code = models.CharField(max_length=32, unique=True, choices=PLAN_CHOICES, default="default")
    name = models.CharField(max_length=64, default="Default Plan")
    max_live_portfolios = models.PositiveIntegerField(default=1)
    max_paper_portfolios = models.PositiveIntegerField(default=3)
    max_broker_accounts_per_portfolio = models.PositiveIntegerField(default=5)

    def __str__(self):
        return f"{self.name} ({self.code})"


class User(AbstractUser):
    email = models.EmailField(unique=True)
    timezone = models.CharField(max_length=64, default="America/Bogota")
    plan = models.ForeignKey(Plan, on_delete=models.SET_NULL, null=True, blank=True, related_name="users")

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self):
        return self.email

    @property
    def effective_plan(self) -> Plan:
        if self.plan:
            return self.plan
        default_plan = Plan.objects.filter(code="default").first()
        if default_plan:
            return default_plan
        # In-memory fallback if not yet created in DB
        return Plan(
            code="default",
            name="Default Plan",
            max_live_portfolios=1,
            max_paper_portfolios=3,
            max_broker_accounts_per_portfolio=5,
        )
