"""
Estado y métricas del pipeline diario de cierre (`DailyPipelineRun`).

Cada paso (marketdata, signals, orchestration) se envuelve en `pipeline_step`, que marca el paso
`running` → `done`/`failed`, guarda su duración en `timings` y registra el error en `errors`.
Las escrituras sobre el registro del día se hacen releyendo la fila con `select_for_update` y
mezclando solo las claves propias, para que pasos concurrentes no se pisen los JSON.
"""
import logging
import time
import traceback
from contextlib import contextmanager
from datetime import date
from typing import Callable, Dict, Generator, Optional

from django.db import transaction
from django.utils import timezone

from apps.trading.models import DailyPipelineRun
from core.market_time import get_ny_now

logger = logging.getLogger(__name__)


def get_or_create_today_pipeline(trade_date: Optional[date] = None) -> DailyPipelineRun:
    """Devuelve (creándolo si falta) el registro del pipeline del día de mercado NY."""
    trade_date = trade_date or get_ny_now().date()
    run, _ = DailyPipelineRun.objects.get_or_create(trade_date=trade_date)
    return run


def update_pipeline(trade_date: date, mutate: Callable[[DailyPipelineRun], None]) -> DailyPipelineRun:
    """Aplica `mutate` sobre la fila del día bloqueada (lectura-modificación-escritura atómica)."""
    get_or_create_today_pipeline(trade_date)
    with transaction.atomic():
        run = DailyPipelineRun.objects.select_for_update().get(trade_date=trade_date)
        mutate(run)
        run.save()
    return run


def record_timing(key: str, metrics: Dict, trade_date: Optional[date] = None) -> None:
    """Guarda métricas sueltas (p. ej. del despacho) en `timings[key]` sin tocar el estado de los pasos."""
    trade_date = trade_date or get_ny_now().date()

    def mutate(run):
        timings = dict(run.timings or {})
        timings[key] = metrics
        run.timings = timings

    update_pipeline(trade_date, mutate)


def _refresh_overall_status(run: DailyPipelineRun) -> None:
    """Si los tres pasos terminaron bien (p. ej. gracias a un fallback), el día queda `done`."""
    if run.status != "running" and all(run.step_status(s) == "done" for s in DailyPipelineRun.STEPS):
        run.status = "done"
        run.finished_at = run.finished_at or timezone.now()


@contextmanager
def pipeline_step(step: str, trade_date: Optional[date] = None) -> Generator[Dict, None, None]:
    """
    Marca un paso del pipeline como `running` y, al salir, como `done` (o `failed` si hubo
    excepción, que se re-lanza). Entrega un dict donde el paso deja sus métricas propias
    (instrumentos, versiones, duración por bot...), que se guardan junto a `seconds`.
    """
    assert step in DailyPipelineRun.STEPS, step
    trade_date = trade_date or get_ny_now().date()
    status_field = f"{step}_status"

    def mark_running(run):
        setattr(run, status_field, "running")

    update_pipeline(trade_date, mark_running)
    metrics: Dict = {}
    started = time.monotonic()
    error = None
    try:
        yield metrics
    except Exception as e:
        error = e
        raise
    finally:
        seconds = round(time.monotonic() - started, 3)

        def mark_finished(run):
            setattr(run, status_field, "failed" if error else "done")
            timings = dict(run.timings or {})
            timings[step] = {"seconds": seconds, **metrics}
            run.timings = timings
            if error:
                run.errors = list(run.errors or []) + [{
                    "step": step,
                    "at": timezone.now().isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=5),
                }]
            _refresh_overall_status(run)

        try:
            update_pipeline(trade_date, mark_finished)
        except Exception as e:  # nunca ocultar el resultado del paso por un fallo de métricas
            logger.error(f"No se pudo registrar el paso {step} en DailyPipelineRun: {e}")
