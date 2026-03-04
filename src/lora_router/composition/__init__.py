"""Adapter composition and merging."""

from lora_router.composition.composer import SmartComposer
from lora_router.composition.merger import MergeMethod, merge_adapters

__all__ = ["MergeMethod", "SmartComposer", "merge_adapters"]
