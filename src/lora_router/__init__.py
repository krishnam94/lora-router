"""lora-router: Intelligent LoRA adapter routing, composition, and serving."""

__version__ = "0.1.0"

from lora_router.composition.composer import SmartComposer
from lora_router.composition.merger import MergeMethod, merge_adapters
from lora_router.registry import AdapterRegistry
from lora_router.router import LoRARouter
from lora_router.types import (
    AdapterInfo,
    AdapterSelection,
    ComposerAction,
    MergeConfig,
    RoutingDecision,
)

__all__ = [
    "AdapterInfo",
    "AdapterRegistry",
    "AdapterSelection",
    "ComposerAction",
    "LoRARouter",
    "MergeConfig",
    "MergeMethod",
    "RoutingDecision",
    "SmartComposer",
    "merge_adapters",
]
