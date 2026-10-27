"""
Benchmark de carga del pipeline de cierre (§4.8 de 03_pipeline_cierre_escalable.md).

Clona las versiones live existentes hasta llegar a N bots (≈10 instrumentos cada uno sobre un
universo de ~300 instrumentos únicos: los reales más sintéticos), los asigna a un portafolio paper
y mide los pasos ① marketdata, ② señales y ③ orquestación tal como corren a las 15:50, más un
despacho simulado. Todo ocurre dentro de una transacción que SIEMPRE se revierte: no persiste
señales, runs ni barras, no llama a brokers (sync de Alpaca y envío de órdenes simulados) y no
envía Telegram.
"""
import time
from datetime import datetime, timedelta
from contextlib import ExitStack
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction

from apps.instruments.models import Instrument
from apps.marketdata.providers.fake import FakeProvider
from apps.marketdata.services import MarketDataService
from apps.portfolios.models import Allocation, Portfolio
from apps.strategies.models import Strategy, StrategyInstrument, StrategyVersion
from apps.trading import tasks
from apps.trading.models import OrderIntent, RebalanceRun
from apps.trading.pipeline import get_or_create_today_pipeline

# Si no hay versiones live que clonar, se usan estos bots (engines y parámetros reales del catálogo)
DEFAULT_TEMPLATES = [
    ("single_signal", "one_strategy", {
        "strategies": [{"strategy_name": "Combo_ZCrossDema", "params": [14.0, 90.0, 112.0, 28.0, 3.0, 30.0]}],
        "leverage": 1.0, "max_leverage": 1.5, "use_regimes": True,
    }),
    ("multi_signal", "multi_strategy", {
        "strategies": [
            {"strategy_name": "MeanRev_WeakRSI", "params": [10.0, 30.0, 10.0]},
            {"strategy_name": "TrendFollowing_KVOBullSPT", "params": [7.0, 28.0, 28.0, 2.0]},
        ],
        "leverage": 1.0, "max_leverage": 1.7, "use_regimes": True,
    }),
]
TARGET_SECONDS = 240  # 15:50 → listo antes de 15:54 NY


class Rollback(Exception):
    pass


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class Command(BaseCommand):
    help = "Benchmark del pipeline de cierre con N bots (siempre revierte: no persiste nada ni envía órdenes)."

    def add_arguments(self, parser):
        parser.add_argument("--bots", type=int, default=100)
        parser.add_argument("--instruments", type=int, default=300, help="Instrumentos únicos del universo")
        parser.add_argument("--instruments-per-bot", type=int, default=10)
        parser.add_argument("--no-network", action="store_true", help="Provider falso con latencia simulada")
        parser.add_argument("--latency-ms", type=float, default=700, help="Latencia por request HTTP simulada")
        parser.add_argument("--history-years", type=int, default=25, help="Años de historial de los instrumentos sintéticos")
        parser.add_argument(
            "--dry-run", action="store_true", default=True,
            help="Corre en una transacción que se revierte (único modo soportado; se acepta por compatibilidad)",
        )

    def handle(self, *args, **opts):
        self.opts = opts
        self.results = {}
        try:
            with transaction.atomic():
                with ExitStack() as stack:
                    self._patch_side_effects(stack)
                    setup_seconds = self._setup()
                    self._run_steps()
                    self.results["setup_seconds"] = setup_seconds
                raise Rollback()
        except Rollback:
            pass
        self._report()

    # --- preparación (no se mide) ---

    def _patch_side_effects(self, stack: ExitStack):
        o = self.opts
        if o["no_network"]:
            provider = lambda *a, **k: FakeProvider(latency_ms=o["latency_ms"], history_years=o["history_years"])  # noqa: E731
            stack.enter_context(patch("apps.marketdata.services.YahooProvider", provider))
        # Sin Redis local, cada lock pagaría el intento de conexión: se detecta una vez y se usa el
        # modo local (en producción Redis responde y el coste del lock es despreciable).
        from core import locks
        try:
            locks._redis_client().ping()
            self.results["redis"] = "disponible"
        except Exception:
            stack.enter_context(patch("core.locks._redis_client", side_effect=ConnectionError("redis no disponible")))
            self.results["redis"] = "no disponible (locks en modo local)"
        stack.enter_context(patch("apps.trading.tasks.sync_alpaca_account", return_value={"success": False}))
        stack.enter_context(patch("apps.notifications.telegram.send_to_telegram", return_value=True))
        stack.enter_context(patch("apps.trading.execution.send_to_telegram", return_value=True))

    def _setup(self) -> float:
        o = self.opts
        started = time.monotonic()
        templates = [
            (v.strategy.engine, v.strategy.kind, v.params or {})
            for v in StrategyVersion.objects.filter(status="live", strategy__is_active=True).select_related("strategy")
            if v.strategy.engine != "ranked_allocation"  # depende de fundamentales reales
        ] or DEFAULT_TEMPLATES
        existing = StrategyVersion.objects.filter(status="live", strategy__is_active=True).count()

        # Universo: instrumentos reales + sintéticos hasta --instruments
        pool = list(Instrument.objects.filter(tradable=True).order_by("id")[: o["instruments"]])
        for i in range(len(pool), o["instruments"]):
            pool.append(Instrument.objects.create(symbol=f"BMK{i:04d}", name=f"Benchmark {i}", asset_class="equity"))

        # Estado de las 15:50: historial completo ya descargado por prewarm (sin latencia, no se mide)
        warm = MarketDataService(provider=FakeProvider(history_years=o["history_years"])) if o["no_network"] else MarketDataService()
        for inst in pool:
            if not inst.full_history_synced:
                warm.ensure_history(inst, incremental=False)

        User = get_user_model()
        user = User.objects.create(email="benchmark@local.test", username="benchmark-pipeline")
        portfolio = Portfolio.objects.create(
            user=user, name="Benchmark", kind="paper", paper_initial_capital=Decimal("1000000"),
        )
        to_create = max(0, o["bots"] - existing)
        weight = (Decimal("1") / Decimal(max(to_create, 1))).quantize(Decimal("0.0001"), rounding="ROUND_DOWN")
        per_bot = o["instruments_per_bot"]
        for b in range(to_create):
            engine, kind, params = templates[b % len(templates)]
            strategy = Strategy.objects.create(name=f"bench-{b}", slug=f"bench-{b}", kind=kind, engine=engine)
            version = StrategyVersion.objects.create(strategy=strategy, version=1, status="live", params=params)
            # Bloques contiguos que dan la vuelta al pool: cubren todo el universo y cada
            # instrumento lo comparten varios bots (como QQQ hoy)
            chosen = {pool[(b * per_bot + j) % len(pool)].id: pool[(b * per_bot + j) % len(pool)] for j in range(per_bot)}
            if kind == "cross_asset":
                source, *traded = list(chosen.values())
                StrategyInstrument.objects.create(version=version, instrument=source, role="signal_source")
            else:
                traded = list(chosen.values())
            for inst in traded:
                StrategyInstrument.objects.create(version=version, instrument=inst, role="traded", weight=Decimal("1"))
            Allocation.objects.create(portfolio=portfolio, strategy=strategy, version=version, target_weight=weight)

        self.results["bots"] = existing + to_create
        self.results["universe"] = len({
            i for i in StrategyInstrument.objects.filter(version__status="live", version__strategy__is_active=True)
            .values_list("instrument_id", flat=True)
        })
        return time.monotonic() - started

    # --- pasos medidos ---

    def _timed(self, name, fn):
        started = time.monotonic()
        error = None
        try:
            result = fn()
        except Exception as e:  # el pipeline real sigue con el siguiente paso
            result, error = None, f"{type(e).__name__}: {e}"
        self.results.setdefault("steps", []).append((name, time.monotonic() - started, result, error))

    def _run_steps(self):
        self._timed("1. marketdata", tasks._marketdata_refresh)
        self._timed("2. senales", tasks._compute_regimes_and_signals)
        self._timed("3. orquestacion", tasks._orchestrate_portfolio_runs)

        timings = get_or_create_today_pipeline().timings or {}
        self.results["per_bot"] = list((timings.get("signals") or {}).get("per_version", {}).values())
        self.results["marketdata_metrics"] = {k: v for k, v in (timings.get("marketdata") or {}).items() if k != "error_symbols"}

        # Despacho simulado: una "llamada al broker" con la misma latencia por intent, en serie como hoy
        intent_ids = list(
            OrderIntent.objects.filter(run__status="executing", status="pending").values_list("id", flat=True)
        )
        latency = self.opts["latency_ms"] / 1000

        def fake_dispatch(dispatcher, intent):
            time.sleep(latency)
            intent.status = "dispatched"
            intent.save(update_fields=["status"])

        def dispatch():
            from apps.trading.execution import OrderDispatcher
            with patch.object(OrderDispatcher, "dispatch_intent", fake_dispatch):
                dispatcher = OrderDispatcher()
                return sum(1 for i in intent_ids if tasks.dispatch_pending_intent(dispatcher, i))

        self._timed("despacho (simulado)", dispatch)
        self.results["intents"] = len(intent_ids)
        self.results["runs"] = RebalanceRun.objects.filter(portfolio__name="Benchmark").count()

    # --- reporte ---

    def _report(self):
        r = self.results
        o = self.opts
        w = self.stdout.write
        w("")
        w(f"Benchmark pipeline: {r['bots']} bots, universo {r['universe']} instrumentos, "
          f"{'sin red, latencia ' + str(int(o['latency_ms'])) + ' ms/request' if o['no_network'] else 'red real'}")
        w(f"(preparacion no medida: {r['setup_seconds']:.1f} s; Redis {r['redis']}; todo revertido al terminar)")
        w("")
        w(f"{'Paso':<22}{'Duracion (s)':>14}  Resultado")
        w("-" * 78)
        pipeline_total = 0.0
        for name, seconds, result, error in r["steps"]:
            if not name.startswith("despacho"):
                pipeline_total += seconds
            w(f"{name:<22}{seconds:>14.1f}  {error or result}")
        w("-" * 78)
        ready_at = (datetime(2000, 1, 1, 15, 50) + timedelta(seconds=pipeline_total)).strftime("%H:%M:%S")
        w(f"{'Pipeline 1+2+3':<22}{pipeline_total:>14.1f}  objetivo <= {TARGET_SECONDS} s "
          f"(listo a las {ready_at} NY) "
          f"-> {'CUMPLE' if pipeline_total <= TARGET_SECONDS else 'NO CUMPLE'}")
        per_bot = r["per_bot"]
        if per_bot:
            w(f"Senales por bot: p50 {_percentile(per_bot, 50):.3f} s  |  p95 {_percentile(per_bot, 95):.3f} s  |  "
              f"max {max(per_bot):.3f} s  |  n={len(per_bot)}")
        if r.get("marketdata_metrics"):
            w(f"Marketdata: {r['marketdata_metrics']}")
        w(f"Orquestacion: {r['runs']} run(s) del portafolio benchmark, {r['intents']} intents a despachar")
