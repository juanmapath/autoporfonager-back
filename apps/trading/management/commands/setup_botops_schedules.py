from django.core.management.base import BaseCommand
from django_celery_beat.models import PeriodicTask, CrontabSchedule
import json


class Command(BaseCommand):
    help = "Sets up automated trading and signal schedules in django-celery-beat."

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Configuring Automated Algorithmic Trading Schedules (America/New_York)..."))

        # Timezone: America/New_York
        tz = "America/New_York"

        # Clean up legacy tasks if present
        PeriodicTask.objects.filter(name__startswith="[1/5]").delete()
        PeriodicTask.objects.filter(name__startswith="[2/5]").delete()
        PeriodicTask.objects.filter(name__startswith="[3/5]").delete()
        PeriodicTask.objects.filter(name__startswith="[4/5]").delete()
        PeriodicTask.objects.filter(name__startswith="[5/5]").delete()

        schedules_def = [
            {
                "name": "[1/6] Daily Algorithmic Trading Pipeline",
                "task": "run_daily_trading_pipeline",
                "minute": "50",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Pipeline consecutivo: descarga market data, genera señales, rebalancea carteras y despacha órdenes MOC/Market (15:50 NY).",
            },
            {
                "name": "[2/6] Fallback: Compute Quantitative Signals",
                "task": "compute_regimes_and_signals",
                "minute": "52",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Respaldo automático a las 15:52 NY en caso de retraso o fallo en la descarga previa.",
            },
            {
                "name": "[3/6] Fallback: Portfolio Rebalance & Allocation Orchestration",
                "task": "orchestrate_portfolio_runs",
                "minute": "53",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Respaldo automático a las 15:53 NY: genera rebalanceos para cualquier cartera pendiente hoy.",
            },
            {
                "name": "[4/6] Fallback: Order Dispatch (Alpaca & Simulated)",
                "task": "dispatch_moc_and_manual_orders",
                "minute": "55",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Respaldo final de despacho a las 15:55 NY antes del cierre de mercado para cualquier orden pendiente.",
            },
            {
                "name": "[5/6] Post-Close Broker Fills & Trade Reconciliation",
                "task": "reconcile_broker_fills",
                "minute": "05",
                "hour": "16",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Consulta a Alpaca tras el cierre (16:05 NY) los fills, precios de ejecución reales y actualiza posiciones contables.",
            },
            {
                "name": "[6/6] Post-Market Daily Equity & Position Snapshot",
                "task": "accounting_daily_snapshot",
                "minute": "15",
                "hour": "16",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Snapshot contable oficial de cierre del AUM total, cash y posiciones (16:15 NY).",
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
