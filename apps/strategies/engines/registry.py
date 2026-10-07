from typing import Dict, Type
from apps.strategies.engines.base import BaseEngine
from apps.strategies.engines.single_signal import SingleSignalEngine
from apps.strategies.engines.multi_signal import MultiSignalEngine
from apps.strategies.engines.cross_asset import CrossAssetEngine, FollowPriceEngine


ENGINE_REGISTRY: Dict[str, Type[BaseEngine]] = {
    "single_signal": SingleSignalEngine,
    "multi_signal": MultiSignalEngine,
    "cross_asset": CrossAssetEngine,
    "follow_price": FollowPriceEngine,
}


def get_engine(engine_name: str) -> BaseEngine:
    engine_cls = ENGINE_REGISTRY.get(engine_name)
    if not engine_cls:
        # Default fallback to single_signal
        return SingleSignalEngine()
    return engine_cls()
