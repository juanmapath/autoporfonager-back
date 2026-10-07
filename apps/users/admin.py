from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, Plan


@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "max_live_portfolios", "max_paper_portfolios", "max_broker_accounts_per_portfolio", "created_at")
    search_fields = ("code", "name")


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ("email", "username", "is_staff", "is_superuser", "plan", "timezone", "date_joined")
    list_filter = ("is_staff", "is_superuser", "plan")
    search_fields = ("email", "username")
    ordering = ("email",)

    fieldsets = BaseUserAdmin.fieldsets + (
        ("Portfonager Settings", {"fields": ("timezone", "plan")}),
    )
