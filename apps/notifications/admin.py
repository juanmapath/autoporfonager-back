from django.contrib import admin
from .models import NotificationChannel


@admin.register(NotificationChannel)
class NotificationChannelAdmin(admin.ModelAdmin):
    list_display = ("name", "telegram_chat_id", "is_active", "created_at")
    list_filter = ("is_active",)
