from django.core.management.base import BaseCommand
from django_celery_beat.models import PeriodicTask, CrontabSchedule
import json


class Command(BaseCommand):
    help = "Sets up automated trading and signal schedules in django-celery-beat."

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Configuring Automated Algorithmic Trading Schedules (America/New_York)..."))

        # Timezone: America/New_York
        tz = "America/New_York"

        schedules_def = [
            {
                "name": "[1/5] Market Data Sync & Quotes",
                "task": "marketdata_refresh",
                "minute": "30",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Descarga barras OHLCV y cotizaciones más recientes de activos operables (15:30 NY).",
            },
            {
                "name": "[2/5] Compute Macro Regimes & Quantitative Signals",
                "task": "compute_regimes_and_signals",
                "minute": "35",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Calcula regímenes e indicadores cuantitativos 25m antes del cierre (15:35 NY).",
            },
            {
                "name": "[3/5] Portfolio Rebalance & Allocation Orchestration",
                "task": "orchestrate_portfolio_runs",
                "minute": "36",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Evalúa tolerancias de carteras, calcula pesos y genera órdenes target (15:36 NY).",
            },
            {
                "name": "[4/5] Order Dispatch (Alpaca MOC Cutoff & Simulation)",
                "task": "dispatch_moc_and_manual_orders",
                "minute": "48",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Despacha órdenes MOC a Alpaca y genera fills en simulador (15:48 NY).",
            },
            {
                "name": "[5/5] Post-Market Daily Equity & Position Snapshot",
                "task": "accounting_daily_snapshot",
                "minute": "15",
                "hour": "16",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Genera snapshot contable del AUM total y posiciones cerradas (16:15 NY).",
            },
        ]

        for s in schedules_def:
            crontab, _ = CrontabSchedule.objects.get_or_create(
                minute=s["minute"],
                hour=s["hour"],
                day_of_week=s["day_of_week"],
                day_of_month="*",
                month_of_year="*",
                timezone=tz,
            )

            task, created = PeriodicTask.objects.update_or_create(
                name=s["name"],
                defaults={
                    "crontab": crontab,
                    "task": s["task"],
                    "enabled": True,
                    "description": s["description"],
                }
            )

            status_str = "Created" if created else "Updated"
            self.stdout.write(self.style.SUCCESS(f"  [OK] [{status_str}] {s['name']} -> {s['hour']}:{s['minute']} NY (Mon-Fri)"))

        self.stdout.write(self.style.SUCCESS("\nAll automated schedules successfully registered in Celery Beat database."))
