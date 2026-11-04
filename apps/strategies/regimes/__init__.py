from .base import RegimeProvider
from .macro import MacroRegimeProvider

# Registro de proveedores. Siguientes (fuera de la fase actual): "vol", "trend", "hmm".
REGIME_PROVIDERS = {"macro": MacroRegimeProvider()}

__all__ = ["RegimeProvider", "MacroRegimeProvider", "REGIME_PROVIDERS"]
