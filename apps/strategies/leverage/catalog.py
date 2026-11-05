"""Catálogo de tácticas de apalancamiento para el front (políticas, capas y proveedores de régimen)."""
from apps.strategies.leverage.layers import layer_catalog
from apps.strategies.regimes import REGIME_PROVIDERS

POLICIES = [
    {"key": "fixed", "name": "Fijo", "needs_regimes": False,
     "description": "Siempre el apalancamiento base."},
    {"key": "regime_pf", "name": "PF por régimen", "default": True,
     "description": "Profit Factor de los trades pasados que entraron en el mismo régimen de mercado que hoy. "
                    "Con menos de min_trades en el régimen, base. Se fija en la entrada.", "needs_regimes": True,
     "defaults": {"min_trades": 3, "lookback_years": 15}},
    {"key": "layered", "name": "Capas combinables",
     "description": "Escaladores (multiplican), topes (min) y políticas de etiqueta: "
                    "L = clip(min(base × Π escaladores, topes), mínimo, máximo). Recalcula a diario por defecto.",
     "needs_regimes": False,
     "defaults": {"apply": "daily", "min": 1.0, "min_change": 0.15, "max_step_up": 0.25}},
]
UPCOMING_PROVIDERS = {"vol": "Volatilidad", "trend": "Tendencia", "hmm": "HMM"}


def leverage_catalog() -> dict:
    providers = [{"key": k, "name": "Macro (crédito, dólar, VIX)" if k == "macro" else k, "available": True}
                 for k in REGIME_PROVIDERS]
    providers += [{"key": k, "name": n, "available": False} for k, n in UPCOMING_PROVIDERS.items()
                  if k not in REGIME_PROVIDERS]
    return {"policies": POLICIES, "layers": layer_catalog(), "providers": providers}
