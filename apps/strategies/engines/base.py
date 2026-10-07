from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
from decimal import Decimal


class BaseEngine(ABC):
    """
    Base Strategy Engine.
    Engines are pure functional calculators: they receive market data and parameters,
    and return target exposures (-N to +N) and diagnostic indicators.
    They NEVER touch broker APIs or place orders directly.
    """

    @abstractmethod
    def compute_signal(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        instruments_weights: Dict[str, Decimal],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
        """
        Computes the target exposure for each traded instrument.
        Returns:
            targets: Dict[symbol, Decimal target_exposure] e.g. {"QQQ": Decimal("1.0")}
            diagnostics: Dict[str, Any] indicator values, regime info, etc.
        """
        pass
