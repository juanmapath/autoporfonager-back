from django.contrib import admin
from .models import KillSwitch, ReconciliationReport, FundingAlert


@admin.register(KillSwitch)
class KillSwitchAdmin(admin.ModelAdmin):
    list_display = ("id", "portfolio", "broker_account", "strategy", "is_active", "activated_at", "reason")
    list_filter = ("is_active", "activated_at")


@admin.register(ReconciliationReport)
class ReconciliationReportAdmin(admin.ModelAdmin):
    list_display = ("broker_account", "date", "matched")
    list_filter = ("matched", "date")


@admin.register(FundingAlert)
class FundingAlertAdmin(admin.ModelAdmin):
    list_display = ("id", "portfolio", "broker_account", "required_cash", "available_cash", "status", "created_at")
    list_filter = ("status",)
