from django.db import models
from core.models import TimeStampedModel
from apps.instruments.models import Instrument


class Bar(TimeStampedModel):
    """Daily OHLCV historical bar for an instrument."""
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, related_name="bars")
    date = models.DateField(db_index=True)
    open = models.DecimalField(max_digits=16, decimal_places=6)
    high = models.DecimalField(max_digits=16, decimal_places=6)
    low = models.DecimalField(max_digits=16, decimal_places=6)
    close = models.DecimalField(max_digits=16, decimal_places=6)
    adj_close = models.DecimalField(max_digits=16, decimal_places=6)
    volume = models.DecimalField(max_digits=20, decimal_places=2, default=0)

    class Meta:
        unique_together = ("instrument", "date")
        ordering = ["date"]

    def __str__(self):
        return f"{self.instrument.symbol} - {self.date}: {self.close}"


class LatestQuote(TimeStampedModel):
    """Latest real-time or decision-window price snapshot."""
    instrument = models.OneToOneField(Instrument, on_delete=models.CASCADE, related_name="latest_quote")
    price = models.DecimalField(max_digits=16, decimal_places=6)
    timestamp = models.DateTimeField(db_index=True)
    currency = models.CharField(max_length=8, default="USD")

    def __str__(self):
        return f"{self.instrument.symbol} @ {self.price} ({self.timestamp})"
