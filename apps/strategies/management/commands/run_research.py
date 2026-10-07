from django.core.management.base import BaseCommand
import pandas as pd
from core.security_guard import is_local_mode_active
from apps.strategies.research.runner import LocalResearchRunner


class Command(BaseCommand):
    help = "Execute exclusive local research analysis (parameter optimization, regime matrix)."

    def handle(self, *args, **options):
        if not is_local_mode_active():
            self.stdout.write(self.style.ERROR("ERROR: run_research is an EXCLUSIVE local script and cannot run when LOCAL_MODE is False."))
            return

        self.stdout.write(self.style.SUCCESS("LOCAL_MODE is ACTIVE: Running local research analysis safely without touching client databases."))
        
        runner = LocalResearchRunner()
        # Generate sample price series for research grid test
        dates = pd.date_range("2023-01-01", "2024-01-01", freq="B")
        prices = 100 + (pd.Series(range(len(dates))) * 0.1)
        df = pd.DataFrame({
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": 1000000.0,
        }, index=dates)

        grid = [
            [15, 1.5],
            [20, 2.0],
            [25, 2.5],
        ]
        res = runner.run_parameter_grid(df, "bollinger_bands", grid)
        self.stdout.write(self.style.SUCCESS(f"Grid search completed. Evaluated {len(res)} combinations."))
        self.stdout.write(str(res))
