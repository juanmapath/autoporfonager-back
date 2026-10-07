from django.contrib import admin
from .models import Instrument


@admin.register(Instrument)
class InstrumentAdmin(admin.ModelAdmin):
    list_display = ("symbol", "name", "asset_class", "tradable", "fractionable", "shortable", "yahoo_symbol", "alpaca_symbol")
    list_filter = ("asset_class", "tradable", "fractionable")
    search_fields = ("symbol", "name", "yahoo_symbol", "alpaca_symbol")
