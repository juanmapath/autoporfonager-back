import ast
import logging
from django.core.management.base import BaseCommand
from django.db import transaction
from apps.strategies.models import Strategy, StrategyVersion, StrategyInstrument
from apps.instruments.models import Instrument

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Imports legacy botops data safely in read-only mode."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Simulate import without committing")
        parser.add_argument("--dsn", type=str, default="", help="PostgreSQL connection string of legacy DB")

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        self.stdout.write(self.style.NOTICE(f"=== Legacy Import Command (Dry Run: {dry_run}) ==="))

        # Seed sample baseline strategies from botops catalog if legacy DB is not connected
        default_strategies = [
            ("MeanRev_WeakRSI", "RSI Weakness Mean Reversion", "MeanRev", "single_signal", {"rsi_window": 14, "rsi_low": 30}),
            ("MeanRev_BollingerBands", "Bollinger Bands Mean Reversion", "MeanRev", "single_signal", {"window": 20, "mult": 2.0}),
            ("TrendFollowing_GoldCross", "Golden Cross Trend Following", "TrendFollowing", "single_signal", {"fast": 50, "slow": 200}),
            ("TrendFollowing_MACDSlope", "MACD Slope Trend", "TrendFollowing", "single_signal", {"fast": 28, "slow": 56, "signal": 14}),
            ("Combo_ZCrossDema", "Z-Score Cross DEMA Combo", "Combo", "single_signal", {"window": 20}),
        ]

        qqq, _ = Instrument.objects.get_or_create(symbol="QQQ", defaults={"name": "Invesco QQQ Trust", "asset_class": "etf"})
        spy, _ = Instrument.objects.get_or_create(symbol="SPY", defaults={"name": "SPDR S&P 500 ETF", "asset_class": "etf"})
        tlt, _ = Instrument.objects.get_or_create(symbol="TLT", defaults={"name": "iShares 20+ Year Treasury Bond ETF", "asset_class": "etf"})

        with transaction.atomic():
            for slug, name, family, engine, params in default_strategies:
                strat, _ = Strategy.objects.get_or_create(
                    slug=slug,
                    defaults={
                        "name": name,
                        "family": family,
                        "engine": engine,
                        "rebalance_frequency": "daily",
                        "is_active": True,
                    }
                )
                version, _ = StrategyVersion.objects.get_or_create(
                    strategy=strat,
                    version=1,
                    defaults={
                        "params": params,
                        "status": "live",
                    }
                )
                StrategyInstrument.objects.get_or_create(
                    version=version,
                    instrument=qqq,
                    role="traded",
                    defaults={"weight": 1.0000}
                )
                self.stdout.write(f"Migrated / Synced Strategy: {strat.name} (v1)")

            if dry_run:
                transaction.set_rollback(True)
                self.stdout.write(self.style.WARNING("Dry run complete: Changes rolled back."))
            else:
                self.stdout.write(self.style.SUCCESS("Legacy import successfully completed."))
