from django.contrib import admin
from .models import Bar, LatestQuote


@admin.register(Bar)
class BarAdmin(admin.ModelAdmin):
    list_display = ("instrument", "date", "open", "high", "low", "close", "volume")
    list_filter = ("instrument", "date")
    search_fields = ("instrument__symbol",)


@admin.register(LatestQuote)
class LatestQuoteAdmin(admin.ModelAdmin):
    list_display = ("instrument", "price", "currency", "timestamp")
    search_fields = ("instrument__symbol",)
