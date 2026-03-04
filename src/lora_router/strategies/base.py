"""Base strategy abstract class for adapter routing."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lora_router.registry import AdapterRegistry
from lora_router.types import AdapterSelection


class BaseStrategy(ABC):
    """Abstract base class for routing strategies.

    A strategy takes a query and registry, and returns ranked adapter selections
    with confidence scores.
    """

    @property
    def name(self) -> str:
        """Strategy name for logging and metrics."""
        return self.__class__.__name__

    @abstractmethod
    def route(
        self, query: str, registry: AdapterRegistry, top_k: int = 5
    ) -> list[AdapterSelection]:
        """Route a single query to the best adapters.

        Args:
            query: Input text to route.
            registry: Adapter registry with metadata and embeddings.
            top_k: Maximum number of adapters to return.

        Returns:
            List of AdapterSelection sorted by confidence (descending).
        """

    def route_batch(
        self, queries: list[str], registry: AdapterRegistry, top_k: int = 5
    ) -> list[list[AdapterSelection]]:
        """Route a batch of queries. Default: sequential. Override for efficiency.

        Args:
            queries: List of input texts.
            registry: Adapter registry.
            top_k: Maximum adapters per query.

        Returns:
            List of selection lists, one per query.
        """
        return [self.route(q, registry, top_k=top_k) for q in queries]
