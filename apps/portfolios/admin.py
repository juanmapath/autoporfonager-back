from django.contrib import admin
from .models import Portfolio, Allocation, AllocationChange, RecommendedPortfolio, RecommendedAllocation


class AllocationInline(admin.TabularInline):
    model = Allocation
    extra = 1


class RecommendedAllocationInline(admin.TabularInline):
    model = RecommendedAllocation
    extra = 1


@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    list_display = (
        "name", "user", "kind", "status", "rebalance_frequency",
        "rebalance_tolerance", "rebalance_tolerance_mode", "trading_enabled", "created_at"
    )
    list_filter = ("kind", "status", "rebalance_frequency", "rebalance_tolerance_mode", "trading_enabled")
    search_fields = ("name", "user__email", "user__username")
    inlines = [AllocationInline]


@admin.register(Allocation)
class AllocationAdmin(admin.ModelAdmin):
    list_display = ("portfolio", "strategy", "target_weight", "broker_account", "enabled")
    list_filter = ("enabled", "strategy")
    search_fields = ("portfolio__name", "strategy__name")


@admin.register(AllocationChange)
class AllocationChangeAdmin(admin.ModelAdmin):
    list_display = ("portfolio", "strategy", "old_weight", "new_weight", "source", "effective_at", "applied")
    list_filter = ("source", "applied")
    search_fields = ("portfolio__name", "strategy__name")


@admin.register(RecommendedPortfolio)
class RecommendedPortfolioAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "risk_level", "rebalance_frequency", "rebalance_tolerance", "is_active", "created_at")
    list_filter = ("risk_level", "rebalance_frequency", "is_active")
    search_fields = ("name", "slug")
    inlines = [RecommendedAllocationInline]
