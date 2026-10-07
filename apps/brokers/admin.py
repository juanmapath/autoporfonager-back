from django.contrib import admin
from .models import BrokerCredential, BrokerAccount


@admin.register(BrokerCredential)
class BrokerCredentialAdmin(admin.ModelAdmin):
    list_display = ("user", "provider", "environment", "key_fingerprint", "created_at")
    list_filter = ("provider", "environment")
    search_fields = ("user__email", "key_fingerprint")


@admin.register(BrokerAccount)
class BrokerAccountAdmin(admin.ModelAdmin):
    list_display = ("display_name", "portfolio", "broker_label", "provider", "environment", "execution_mode", "trading_enabled", "managed_capital")
    list_filter = ("provider", "environment", "execution_mode", "trading_enabled", "broker_label")
    search_fields = ("display_name", "broker_label", "portfolio__name", "portfolio__user__email")
