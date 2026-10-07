from django.contrib import admin
from .models import Strategy, StrategyVersion, StrategyInstrument, Signal, BacktestRun


class StrategyInstrumentInline(admin.TabularInline):
    model = StrategyInstrument
    extra = 1


class StrategyVersionInline(admin.StackedInline):
    model = StrategyVersion
    extra = 0
    show_change_link = True


@admin.register(Strategy)
class StrategyAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "family", "kind", "engine", "is_active", "signal_only", "created_at")
    list_filter = ("family", "kind", "is_active", "signal_only")
    search_fields = ("name", "slug", "family")
    inlines = [StrategyVersionInline]


@admin.register(StrategyVersion)
class StrategyVersionAdmin(admin.ModelAdmin):
    list_display = ("strategy", "version", "status", "created_at")
    list_filter = ("status", "strategy")
    search_fields = ("strategy__name", "strategy__slug")
    inlines = [StrategyInstrumentInline]


@admin.register(StrategyInstrument)
class StrategyInstrumentAdmin(admin.ModelAdmin):
    list_display = ("version", "instrument", "role", "weight")
    list_filter = ("role", "instrument")
    search_fields = ("version__strategy__name", "instrument__symbol")


@admin.register(Signal)
class SignalAdmin(admin.ModelAdmin):
    list_display = ("version", "instrument", "target_exposure", "changed", "as_of")
    list_filter = ("changed", "instrument", "version__strategy")
    search_fields = ("version__strategy__name", "instrument__symbol")
    readonly_fields = ("created_at",)


@admin.register(BacktestRun)
class BacktestRunAdmin(admin.ModelAdmin):
    list_display = ("strategy_slug", "status", "initial_capital", "start_date", "end_date", "created_at")
    list_filter = ("status", "strategy_slug")
    search_fields = ("strategy_slug",)
