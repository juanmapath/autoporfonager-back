from django.db import models
from core.models import TimeStampedModel


class Instrument(TimeStampedModel):
    ASSET_CLASS_CHOICES = (
        ("etf", "ETF"),
        ("equity", "Equity / Stock"),
        ("index", "Index"),
        ("crypto", "Crypto"),
    )
    symbol = models.CharField(max_length=32, unique=True, db_index=True)
    name = models.CharField(max_length=128, blank=True)
    asset_class = models.CharField(max_length=32, choices=ASSET_CLASS_CHOICES, default="etf")
    tradable = models.BooleanField(default=True)
    fractionable = models.BooleanField(default=True)
    shortable = models.BooleanField(default=False)
    yahoo_symbol = models.CharField(max_length=32, blank=True)
    alpaca_symbol = models.CharField(max_length=32, blank=True)

    def save(self, *args, **kwargs):
        if not self.yahoo_symbol:
            self.yahoo_symbol = self.symbol
        if not self.alpaca_symbol:
            self.alpaca_symbol = self.symbol
        super().save(*args, **kwargs)

    def __str__(self):
        return self.symbol
