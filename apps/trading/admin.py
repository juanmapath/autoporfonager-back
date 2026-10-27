from django.contrib import admin
from .models import RebalanceRun, Target, OrderIntent, Order, Fill, FillAllocation, DailyPipelineRun


class TargetInline(admin.TabularInline):
    model = Target
    extra = 0
    readonly_fields = ("strategy", "instrument", "broker_account", "sleeve_capital", "target_qty", "current_qty", "delta_qty", "reason")


class OrderIntentInline(admin.TabularInline):
    model = OrderIntent
    extra = 0
    readonly_fields = ("broker_account", "instrument", "net_delta_qty", "idempotency_key", "status")


@admin.register(RebalanceRun)
class RebalanceRunAdmin(admin.ModelAdmin):
    list_display = ("id", "portfolio", "trigger", "status", "as_of", "equity_base", "created_at")
    list_filter = ("status", "trigger")
    search_fields = ("portfolio__name",)
    inlines = [TargetInline, OrderIntentInline]


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ("client_order_id", "portfolio", "broker_account", "instrument", "side", "qty", "source", "status", "created_at")
    list_filter = ("source", "status", "side")
    search_fields = ("client_order_id", "broker_order_id", "instrument__symbol", "portfolio__name")


@admin.register(Fill)
class FillAdmin(admin.ModelAdmin):
    list_display = ("id", "order", "fill_price", "qty", "commission", "source", "filled_at")
    list_filter = ("source",)
    search_fields = ("order__client_order_id", "order__instrument__symbol")


@admin.register(DailyPipelineRun)
class DailyPipelineRunAdmin(admin.ModelAdmin):
    list_display = ("trade_date", "status", "marketdata_status", "signals_status", "orchestration_status", "started_at", "finished_at")
    list_filter = ("status",)
    readonly_fields = [f.name for f in DailyPipelineRun._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
