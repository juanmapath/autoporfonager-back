from django.db import models
from core.models import TimeStampedModel


class NotificationChannel(TimeStampedModel):
    name = models.CharField(max_length=64)
    telegram_chat_id = models.CharField(max_length=64)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} ({self.telegram_chat_id})"
