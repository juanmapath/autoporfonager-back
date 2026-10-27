"""Universo de instrumentos que el pipeline diario necesita refrescar."""
from django.db.models import Q, QuerySet

from apps.accounting.models import LogicalPosition
from apps.instruments.models import Instrument
from apps.strategies.models import StrategyInstrument


def live_strategy_instrument_ids() -> set:
    """Instrumentos (traded y signal_source) de las versiones live de estrategias activas."""
    return set(
        StrategyInstrument.objects.filter(version__status="live", version__strategy__is_active=True)
        .values_list("instrument_id", flat=True)
    )


def pipeline_universe() -> QuerySet:
    """
    Instrumentos del paso ① (marketdata): los que usa alguna versión live de una estrategia activa
    o los que tienen una LogicalPosition abierta (para valorar y liquidar). Deduplicado; excluye
    instrumentos de alta en el catálogo que ningún bot activo usa.
    """
    ids = live_strategy_instrument_ids()
    ids |= set(LogicalPosition.objects.filter(~Q(qty=0)).values_list("instrument_id", flat=True))
    return Instrument.objects.filter(id__in=ids).order_by("id")
