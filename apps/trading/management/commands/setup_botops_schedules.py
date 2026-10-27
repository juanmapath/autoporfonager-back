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
        # Replaced by the 15:57 dispatch; left enabled it would send orders at 15:55.
        PeriodicTask.objects.filter(name="[4/6] Fallback: Order Dispatch (Alpaca & Simulated)").delete()
        # Fallbacks incondicionales reemplazados por fallback_compute_signals / fallback_orchestrate,
        # que solo corren si el paso de hoy quedó pendiente o falló.
        PeriodicTask.objects.filter(name="[2/6] Fallback: Compute Quantitative Signals").delete()
        PeriodicTask.objects.filter(name="[3/6] Fallback: Portfolio Rebalance & Allocation Orchestration").delete()

        schedules_def = [
            {
                "name": "[Prewarm] Market Data Full History & Fundamentals",
                "task": "prewarm_market_data",
                "minute": "30",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Pre-calentamiento a las 15:30 NY: historial completo de instrumentos nuevos del universo y fundamentales faltantes, fuera de la ventana de cierre.",
            },
            {
                "name": "[1/6] Daily Algorithmic Trading Pipeline",
                "task": "run_daily_trading_pipeline",
                "minute": "50",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Pipeline de preparación (15:50 NY): descarga market data, genera señales y rebalancea carteras. Las órdenes se envían a las 15:57 NY.",
            },
            {
                "name": "[2/6] Conditional Fallback: Compute Quantitative Signals",
                "task": "fallback_compute_signals",
                "minute": "52",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Respaldo a las 15:52 NY: recalcula señales solo si el paso de hoy quedó pendiente o falló (se omite si está hecho o en curso).",
            },
            {
                "name": "[3/6] Conditional Fallback: Portfolio Rebalance & Allocation Orchestration",
                "task": "fallback_orchestrate",
                "minute": "53",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Respaldo a las 15:53 NY: orquesta solo si el paso de hoy quedó pendiente o falló; avisa por Telegram si las señales no están actualizadas.",
            },
            {
                "name": "[Watchdog] Daily Pipeline Completion Check",
                "task": "pipeline_watchdog",
                "minute": "56",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Watchdog a las 15:56 NY: alerta por Telegram si el pipeline de hoy no terminó, con el estado y la duración de cada paso.",
            },
            {
                "name": "[4/6] Order Dispatch (Alpaca Market DAY & Simulated)",
                "task": "dispatch_moc_and_manual_orders",
                "minute": "57",
                "hour": "15",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Envío de órdenes a las 15:57 NY: órdenes de mercado DAY (admiten fracciones) justo antes del cierre.",
            },
            {
                "name": "[Watchdog] Post-Cutoff Missed Orders Check",
                "task": "pipeline_post_cutoff_check",
                "minute": "01",
                "hour": "16",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Chequeo a las 16:01 NY: alerta por Telegram si hoy hay rebalanceos en missed_cutoff o intents sin enviar.",
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
                "name": "[Daily] Broker vs Logical Positions Reconciliation",
                "task": "reconcile_positions_vs_broker",
                "minute": "10",
                "hour": "16",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Compara posiciones lógicas con las posiciones reales en Alpaca (16:10 NY) y alerta por Telegram si difieren.",
            },
            {
                "name": "[6/6] Post-Market Daily Equity & Position Snapshot",
                "task": "accounting_daily_snapshot",
                "minute": "15",
                "hour": "16",
                "day_of_week": "1-5",  # Mon-Fri
                "description": "Snapshot contable oficial de cierre del AUM total, cash y posiciones (16:15 NY).",
            },
            {
                "name": "[Weekly] Fundamentals Data Sync (Ranked Universe)",
                "task": "sync_fundamentals_universe",
                "minute": "00",
                "hour": "02",
                "day_of_week": "0",  # Sunday 02:00 AM NY
                "description": "Descarga semanal de balances, estados de resultados y flujos para activos en bots Ranked Allocation.",
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
