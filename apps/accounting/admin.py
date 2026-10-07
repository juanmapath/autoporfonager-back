from django.contrib import admin
from .models import LogicalPosition, BrokerPositionSnapshot, LedgerEntry, EquitySnapshot, SleevePnlDaily, BenchmarkDaily


@admin.register(LogicalPosition)
class LogicalPositionAdmin(admin.ModelAdmin):
    list_display = ("portfolio", "broker_account", "strategy", "instrument", "qty", "avg_cost", "realized_pnl")
    list_filter = ("portfolio", "strategy", "instrument")
    search_fields = ("portfolio__name", "strategy__name", "instrument__symbol")


@admin.register(BrokerPositionSnapshot)
class BrokerPositionSnapshotAdmin(admin.ModelAdmin):
    list_display = ("broker_account", "instrument", "qty", "snapshot_time")
    list_filter = ("broker_account", "instrument")


@admin.register(LedgerEntry)
class LedgerEntryAdmin(admin.ModelAdmin):
    list_display = ("portfolio", "broker_account", "entry_type", "amount", "currency", "reference", "created_at")
    list_filter = ("entry_type", "currency")
    search_fields = ("portfolio__name", "reference")


@admin.register(EquitySnapshot)
class EquitySnapshotAdmin(admin.ModelAdmin):
    list_display = ("portfolio", "date", "total_equity", "cash_balance", "positions_value", "source")
    list_filter = ("portfolio", "date")


@admin.register(SleevePnlDaily)
class SleevePnlDailyAdmin(admin.ModelAdmin):
    list_display = ("portfolio", "strategy", "date", "sleeve_value", "pnl")
    list_filter = ("portfolio", "strategy", "date")


@admin.register(BenchmarkDaily)
class BenchmarkDailyAdmin(admin.ModelAdmin):
    list_display = ("symbol", "date", "close", "cumulative_return")
    list_filter = ("symbol", "date")
