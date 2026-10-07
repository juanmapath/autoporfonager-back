import json
from datetime import datetime
from django.core.management.base import BaseCommand
import pandas as pd
from apps.instruments.models import Instrument
from apps.marketdata.models import Bar
from apps.strategies.backtest import BacktestEngine
from apps.strategies.signals.catalog import STRATEGY_MAP


class Command(BaseCommand):
    help = "Run local backtest on a strategy with historical bars."

    def add_arguments(self, parser):
        parser.add_argument("--strategy", type=str, default="MeanRev_WeakRSI", help="Strategy function name")
        parser.add_argument("--symbol", type=str, default="QQQ", help="Instrument symbol")
        parser.add_argument("--from", dest="from_date", type=str, default="2022-01-01")
        parser.add_argument("--to", dest="to_date", type=str, default="2026-01-01")
        parser.add_argument("--capital", type=float, default=50000.0)
        parser.add_argument("--params", type=str, default="{}", help="JSON params")

    def handle(self, *args, **options):
        strat_name = options["strategy"]
        symbol = options["symbol"]
        capital = options["capital"]
        params = json.loads(options["params"])
        from_d = options["from_date"]
        to_d = options["to_date"]

        self.stdout.write(self.style.NOTICE(f"=== Starting Backtest: {strat_name} on {symbol} (${capital:,.2f}) ==="))

        # Fetch bars from local DB or generate synthetic for testing
        instrument, _ = Instrument.objects.get_or_create(symbol=symbol)
        bars_qs = Bar.objects.filter(instrument=instrument, date__gte=from_d, date__lte=to_d).order_by("date")

        if bars_qs.count() < 30:
            self.stdout.write(self.style.WARNING("Insufficient bars in DB, generating sample trend bars for testing..."))
            dates = pd.date_range(start=from_d, end=to_d, freq="B")
            prices = 300.0 + (pd.Series(range(len(dates))) * 0.15) + (pd.Series(range(len(dates))).apply(lambda x: (x % 7) - 3))
            df = pd.DataFrame({
                "Open": prices,
                "High": prices + 1.5,
                "Low": prices - 1.5,
                "Close": prices,
                "Volume": 50000000.0,
            }, index=dates)
        else:
            data = []
            for b in bars_qs:
                data.append({
                    "Date": b.date,
                    "Open": float(b.open),
                    "High": float(b.high),
                    "Low": float(b.low),
                    "Close": float(b.close),
                    "Volume": float(b.volume),
                })
            df = pd.DataFrame(data).set_index("Date")

        engine = BacktestEngine(initial_cash=capital)
        if not params and strat_name in STRATEGY_MAP:
            # Default params based on strategy
            if "RSI" in strat_name:
                params = {"rsi_window": 14, "rsi_low": 30, "rsi_high": 70}
            elif "Bollinger" in strat_name:
                params = [20, 2.0]
            else:
                params = {}

        results = engine.run(df, strat_name, params)

        self.stdout.write(self.style.SUCCESS(f"=== Backtest Finished ==="))
        self.stdout.write(f"Final Value:     ${results['final_value']:,.2f}")
        self.stdout.write(f"Total ROI:       {results['roi_pct']:.2f}%")
        self.stdout.write(f"CAGR:            {results['cagr_pct']:.2f}%")
        self.stdout.write(f"Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
        self.stdout.write(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
        self.stdout.write(f"Total Trades:    {results['total_trades']}")
        self.stdout.write(f"Win Rate:        {results['win_rate_pct']:.2f}%")
        self.stdout.write(f"Profit Factor:   {results['profit_factor']}")
