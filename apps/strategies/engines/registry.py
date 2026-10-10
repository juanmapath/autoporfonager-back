from typing import Dict, Type
from apps.strategies.engines.base import BaseEngine
from apps.strategies.engines.single_signal import SingleSignalEngine
from apps.strategies.engines.multi_signal import MultiSignalEngine
from apps.strategies.engines.cross_asset import CrossAssetEngine
from apps.strategies.engines.ranked_allocation import RankedAllocationEngine
from apps.strategies.engines.hold import HoldEngine
from apps.strategies.engines.smart_accumulator import SmartAccumulatorEngine


ENGINE_REGISTRY: Dict[str, Type[BaseEngine]] = {
    "single_signal": SingleSignalEngine,
    "multi_signal": MultiSignalEngine,
    "cross_asset": CrossAssetEngine,
    "ranked_allocation": RankedAllocationEngine,
    "hold": HoldEngine,
    "smart_accumulator": SmartAccumulatorEngine,
}


def get_engine(engine_name: str) -> BaseEngine:
    engine_cls = ENGINE_REGISTRY.get(engine_name)
    if not engine_cls:
        # Default fallback to single_signal
        return SingleSignalEngine()
    return engine_cls()
