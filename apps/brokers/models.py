from django.db import models
from django.conf import settings
from django.core.exceptions import ValidationError
from core.models import TimeStampedModel
from core.crypto import encrypt_json, decrypt_json


class BrokerCredential(TimeStampedModel):
    PROVIDER_CHOICES = (
        ("alpaca", "Alpaca"),
        ("manual", "Manual Broker (eToro, IBKR, etc.)"),
    )
    ENV_CHOICES = (
        ("live", "Live Trading"),
        ("paper", "Paper / Sandbox Trading"),
    )
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="broker_credentials")
    provider = models.CharField(max_length=32, choices=PROVIDER_CHOICES, default="alpaca")
    environment = models.CharField(max_length=16, choices=ENV_CHOICES, default="paper")
    auth_type = models.CharField(max_length=32, default="api_key")
    secret_encrypted = models.BinaryField()
    key_fingerprint = models.CharField(max_length=16, blank=True)

    def set_secrets(self, secrets_dict: dict):
        self.secret_encrypted = encrypt_json(secrets_dict)
        key_id = str(secrets_dict.get("key_id", ""))
        self.key_fingerprint = key_id[-4:] if len(key_id) >= 4 else "••••"

    def get_secrets(self) -> dict:
        if not self.secret_encrypted:
            return {}
        return decrypt_json(bytes(self.secret_encrypted))

    def __str__(self):
        return f"{self.user.email} - {self.provider} ({self.environment}) [***{self.key_fingerprint}]"


class BrokerAccount(TimeStampedModel):
    PROVIDER_CHOICES = (
        ("alpaca", "Alpaca"),
        ("manual", "Manual Broker"),
    )
    ENV_CHOICES = (
        ("live", "Live"),
        ("paper", "Paper"),
    )
    EXECUTION_MODE_CHOICES = (
        ("auto", "Auto (API / MOC)"),
        ("manual", "Manual (Simulated Quote + Telegram alert)"),
    )

    portfolio = models.ForeignKey("portfolios.Portfolio", on_delete=models.CASCADE, related_name="broker_accounts")
    credential = models.ForeignKey(BrokerCredential, on_delete=models.SET_NULL, null=True, blank=True)
    provider = models.CharField(max_length=32, choices=PROVIDER_CHOICES, default="manual")
    environment = models.CharField(max_length=16, choices=ENV_CHOICES, default="live")
    execution_mode = models.CharField(max_length=16, choices=EXECUTION_MODE_CHOICES, default="manual")
    display_name = models.CharField(max_length=128)
    broker_label = models.CharField(max_length=64, default="eToro", help_text="e.g. eToro, Alpaca, Interactive Brokers")
    external_account_id = models.CharField(max_length=64, blank=True)
    managed_capital = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    commission_model = models.JSONField(default=dict, blank=True)
    order_policy = models.JSONField(default=dict, blank=True)
    trading_enabled = models.BooleanField(default=True)

    def clean(self):
        super().clean()
        if self.portfolio_id:
            port = self.portfolio
            if port.kind == "paper" and self.environment != "paper":
                raise ValidationError("Un portafolio Paper solo puede vincularse a cuentas de broker en entorno Paper / Sandbox.")
            if port.kind == "live" and self.environment == "paper":
                raise ValidationError("Un portafolio Live con dinero real no debe vincularse a una cuenta Paper / Sandbox.")
            if self.pk is None:
                plan = port.user.effective_plan
                current_accounts = port.broker_accounts.filter(trading_enabled=True).count()
                if current_accounts >= plan.max_broker_accounts_per_portfolio:
                    raise ValidationError(f"Plan limit exceeded: Maximum {plan.max_broker_accounts_per_portfolio} broker accounts per portfolio.")

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.display_name} ({self.broker_label}) [{self.execution_mode.upper()}]"
