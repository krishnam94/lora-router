"""Data models for lora-router."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AdapterInfo(BaseModel):
    """Metadata for a registered LoRA adapter."""

    name: str = Field(..., description="Unique adapter identifier")
    path: str = Field("", description="Local path or HuggingFace Hub ID")
    description: str = Field("", description="What this adapter is trained for")
    domain: str = Field("", description="Task domain (e.g. 'code', 'math', 'nli')")
    example_queries: list[str] = Field(
        default_factory=list,
        description="Example inputs this adapter handles well",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (training config, metrics, etc.)",
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def has_examples(self) -> bool:
        return len(self.example_queries) > 0

    @property
    def text_for_embedding(self) -> str:
        """Combine description and examples into a single text for embedding."""
        parts = []
        if self.description:
            parts.append(self.description)
        if self.example_queries:
            parts.extend(self.example_queries[:5])
        return " ".join(parts) if parts else self.name


class AdapterSelection(BaseModel):
    """A single adapter selected by a routing strategy."""

    adapter_name: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence 0-1")
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-strategy scores (e.g. {'similarity': 0.85, 'seqr': 0.72})",
    )


class ComposerAction(str, Enum):
    """Action decided by SmartComposer."""

    SINGLE = "single"
    COMPOSE = "compose"
    FALLBACK = "fallback"


class MergeMethod(str, Enum):
    """Adapter composition method."""

    LINEAR = "linear"
    TIES = "ties"
    DARE = "dare"
    CAT = "cat"


class MergeConfig(BaseModel):
    """Configuration for adapter composition."""

    method: MergeMethod = MergeMethod.LINEAR
    top_k: int = Field(3, ge=1, description="Max adapters to compose")
    threshold_high: float = Field(
        0.8, ge=0.0, le=1.0, description="Above this: use single adapter"
    )
    threshold_low: float = Field(
        0.3, ge=0.0, le=1.0, description="Below this: fallback to base model"
    )
    density: float = Field(
        0.5, ge=0.0, le=1.0, description="DARE/TIES density parameter"
    )


class RoutingDecision(BaseModel):
    """Complete routing decision with metadata."""

    selections: list[AdapterSelection] = Field(
        ..., description="Ranked adapter selections"
    )
    action: ComposerAction = Field(
        ComposerAction.SINGLE, description="Composer action taken"
    )
    strategy_used: str = Field("", description="Name of routing strategy")
    composition_method: str = Field("", description="Merge method if composed")
    latency_ms: float = Field(0.0, ge=0.0, description="Routing latency in ms")
    query: str = Field("", description="Original query text")

    @property
    def top_adapter(self) -> str | None:
        """Name of the highest-confidence adapter."""
        return self.selections[0].adapter_name if self.selections else None

    @property
    def top_confidence(self) -> float:
        """Confidence of the top selection."""
        return self.selections[0].confidence if self.selections else 0.0

    @property
    def adapter_names(self) -> list[str]:
        """All selected adapter names in ranked order."""
        return [s.adapter_name for s in self.selections]
