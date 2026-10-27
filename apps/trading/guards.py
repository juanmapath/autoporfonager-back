"""Guards de las tareas de trading: solo corren en días hábiles NYSE."""
import functools
import logging

from core.market_time import get_ny_now, is_trading_day

logger = logging.getLogger(__name__)

NOT_TRADING_DAY = "Skipped: not a trading day"


def mark_pipeline_skipped(trade_date) -> None:
    """Si ya existe el DailyPipelineRun del día, marca como `skipped` el día y los pasos sin terminar."""
    from apps.trading.models import DailyPipelineRun

    run = DailyPipelineRun.objects.filter(trade_date=trade_date).first()
    if not run:
        return
    run.status = "skipped"
    for step in DailyPipelineRun.STEPS:
        if run.step_status(step) in ("pending", "running"):
            setattr(run, f"{step}_status", "skipped")
    run.save()


def skip_if_not_trading_day(task_name: str):
    """
    Decorador para tareas de trading: en un día no hábil NYSE (feriado o fin de semana) no ejecuta
    la tarea, deja log, marca el DailyPipelineRun del día como `skipped` si existe y devuelve
    NOT_TRADING_DAY. Se aplica debajo de @shared_task para conservar el `name=` de la tarea.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            today = get_ny_now().date()
            if not is_trading_day(today):
                logger.info(f"[{task_name}] {today} no es día hábil NYSE. Omitiendo.")
                mark_pipeline_skipped(today)
                return NOT_TRADING_DAY
            return func(*args, **kwargs)
        return wrapper
    return decorator
