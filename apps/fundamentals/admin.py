from django.contrib import admin
from apps.fundamentals.models import FundamentalStatement, FundamentalFetchLog


@admin.register(FundamentalStatement)
class FundamentalStatementAdmin(admin.ModelAdmin):
    list_display = ("instrument", "period_type", "period_end", "available_at", "source", "updated_at")
    list_filter = ("period_type", "source")
    search_fields = ("instrument__symbol",)


@admin.register(FundamentalFetchLog)
class FundamentalFetchLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "source", "instrument", "status", "calls_used", "statements_saved")
    list_filter = ("source", "status")
