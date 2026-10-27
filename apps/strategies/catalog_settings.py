"""
Disponibilidad de las estrategias del catálogo para el armado de bots.

El catálogo vive en código (signals/metadata.py); el bloqueo administrable vive en BD
(StrategyCatalogSetting). El backtest no consulta esto: siempre puede usar todo el catálogo.
"""
from typing import Any, Dict, Set

from apps.strategies.models import StrategyCatalogSetting
from apps.strategies.signals.metadata import STRATEGIES_CATALOG

SHORT_BLOCKED_REASON = "Estrategia en corto: los motores solo operan en largo."


def strategies_catalog_with_availability() -> Dict[str, Dict[str, Any]]:
    """Catálogo completo con `direction`, `available_for_bots`, `locked` y `blocked_reason` por estrategia."""
    overrides = {s.strategy_name: s for s in StrategyCatalogSetting.objects.all()}
    catalog = {}
    for key, definition in STRATEGIES_CATALOG.items():
        direction = definition.get("direction", "long")
        override = overrides.get(key)
        if direction != "long":
            # No se puede habilitar desde el admin hasta que los motores soporten cortos
            available, locked, reason = False, True, SHORT_BLOCKED_REASON
        elif override is not None:
            available, locked = override.available_for_bots, False
            reason = "" if available else (override.note or "Bloqueada por un administrador.")
        else:
            available, locked, reason = True, False, ""
        catalog[key] = {
            **definition,
            "direction": direction,
            "available_for_bots": available,
            "locked": locked,
            "blocked_reason": reason,
            "note": override.note if override else "",
        }
    return catalog


def blocked_strategy_names() -> Set[str]:
    return {k for k, v in strategies_catalog_with_availability().items() if not v["available_for_bots"]}
