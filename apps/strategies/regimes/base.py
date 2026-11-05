"""Contrato de los proveedores de régimen de mercado."""
from typing import Any, Dict, List, Protocol

import pandas as pd


class RegimeProvider(Protocol):
    key: str  # "macro", "vol", "trend", "hmm"

    def required_symbols(self) -> List[str]:
        """Símbolos cuyas barras diarias necesita `compute` (instrumentos no operables con historial completo)."""

    def compute(self, bars: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> pd.Series:
        """
        Etiqueta de régimen por fecha de mercado (index = fechas, valores str o None cuando no hay
        datos suficientes). CAUSAL: la etiqueta de la fecha t solo depende de datos hasta t.
        """
