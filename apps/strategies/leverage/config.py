"""Configuración de la política `layered` (04e §4) y validación con mensajes claros."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from apps.strategies.leverage.layers import LAYERS

APPLY_MODES = ("entry", "daily")
LIST_KIND = {"scalers": "scaler", "caps": "cap", "label_policies": "label"}


@dataclass(frozen=True)
class LayerRef:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self):
        return LAYERS[self.key]

    def resolved_params(self) -> Dict[str, Any]:
        return {**self.spec.defaults, **self.params}

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "params": dict(self.params)}


@dataclass(frozen=True)
class LayeredConfig:
    base: float = 1.0
    min: float = 1.0
    max: float = 2.0
    apply: str = "daily"
    min_change: float = 0.15
    max_step_up: float = 0.25
    step_days: int = 5
    scalers: Tuple[LayerRef, ...] = ()
    caps: Tuple[LayerRef, ...] = ()
    label_policies: Tuple[LayerRef, ...] = ()
    combine_labels: str = "min"
    provider: str = "macro"

    @property
    def layers(self) -> Tuple[LayerRef, ...]:
        return self.scalers + self.caps + self.label_policies

    @property
    def needs_regimes(self) -> bool:
        return any(l.spec.needs_regimes for l in self.layers)

    def n_layers(self) -> int:
        return len(self.layers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "layered", "base": self.base, "min": self.min, "max": self.max, "apply": self.apply,
            "min_change": self.min_change, "max_step_up": self.max_step_up,
            "scalers": [l.to_dict() for l in self.scalers], "caps": [l.to_dict() for l in self.caps],
            "label_policies": [l.to_dict() for l in self.label_policies], "combine_labels": self.combine_labels,
        }


def _refs(raw: Any, list_name: str) -> Tuple[LayerRef, ...]:
    if raw in (None, ""):
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"leverage_policy.{list_name} debe ser una lista.")
    expected = LIST_KIND[list_name]
    out: List[LayerRef] = []
    for item in raw:
        if isinstance(item, str):
            item = {"key": item}
        if not isinstance(item, Mapping) or not item.get("key"):
            raise ValueError(f"leverage_policy.{list_name}: cada capa necesita 'key'.")
        key = str(item["key"])
        spec = LAYERS.get(key)
        if spec is None:
            raise ValueError(f"Capa de apalancamiento inexistente: {key!r}. Disponibles: {', '.join(LAYERS)}.")
        if spec.kind != expected:
            raise ValueError(
                f"La capa {key!r} es de tipo '{spec.kind}' y no puede ir en '{list_name}' (espera '{expected}')."
            )
        params = item.get("params") or {}
        if not isinstance(params, Mapping):
            raise ValueError(f"leverage_policy.{list_name}[{key}].params debe ser un objeto.")
        out.append(LayerRef(key, dict(params)))
    return tuple(out)


def parse_layered(raw: Mapping[str, Any], fallback_base: float = 1.0, fallback_max: float = 1.0) -> LayeredConfig:
    def num(name, default):
        v = raw.get(name, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"leverage_policy.{name} debe ser numérico.")

    base = num("base", fallback_base)
    lmax = num("max", max(fallback_max, base))
    lmin = num("min", 1.0)
    if lmin > lmax:
        raise ValueError(f"leverage_policy: min ({lmin}) no puede ser mayor que max ({lmax}).")
    if base <= 0:
        raise ValueError("leverage_policy.base debe ser > 0.")
    apply = str(raw.get("apply") or "daily")
    if apply not in APPLY_MODES:
        raise ValueError(f"leverage_policy.apply invalido: {apply!r}. Opciones: {', '.join(APPLY_MODES)}.")
    combine = str(raw.get("combine_labels") or "min")
    if combine not in ("min", "mean"):
        raise ValueError(f"leverage_policy.combine_labels invalido: {combine!r}. Opciones: min, mean.")
    return LayeredConfig(
        base=base, min=lmin, max=lmax, apply=apply,
        min_change=num("min_change", 0.15), max_step_up=num("max_step_up", 0.25),
        step_days=int(raw.get("step_days", 5)),
        scalers=_refs(raw.get("scalers"), "scalers"), caps=_refs(raw.get("caps"), "caps"),
        label_policies=_refs(raw.get("label_policies"), "label_policies"),
        combine_labels=combine, provider=str(raw.get("provider") or "macro"),
    )


def required_symbols(cfg: Optional[LayeredConfig]) -> List[str]:
    """Símbolos de referencia (no operables) que necesitan las capas, sin el activo propio."""
    if cfg is None:
        return []
    out: List[str] = []
    for ref in cfg.layers:
        for s in ref.spec.symbols(ref.resolved_params()):
            if s not in out:
                out.append(s)
    return out
