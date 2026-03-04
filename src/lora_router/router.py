"""Main LoRARouter interface."""

from __future__ import annotations

import time

from lora_router.composition.composer import SmartComposer
from lora_router.registry import AdapterRegistry
from lora_router.strategies.base import BaseStrategy
from lora_router.types import MergeConfig, RoutingDecision


class LoRARouter:
    """Main router that combines a strategy with smart composition.

    Usage:
        registry = AdapterRegistry(embedder=encoder)
        registry.register(AdapterInfo(name="code", ...))
        strategy = SimilarityStrategy(encoder=encoder)
        router = LoRARouter(registry, strategy)
        decision = router.route("Write a Python function to sort a list")
        print(decision.top_adapter, decision.top_confidence)

    Args:
        registry: Adapter registry with registered adapters.
        strategy: Routing strategy to use.
        merge_config: Configuration for smart composition.
        composer: Optional custom SmartComposer. If None, creates one from merge_config.
    """

    def __init__(
        self,
        registry: AdapterRegistry,
        strategy: BaseStrategy,
        merge_config: MergeConfig | None = None,
        composer: SmartComposer | None = None,
    ) -> None:
        self.registry = registry
        self.strategy = strategy
        self.merge_config = merge_config or MergeConfig()
        self.composer = composer or SmartComposer(self.merge_config)

    def route(self, query: str, top_k: int | None = None) -> RoutingDecision:
        """Route a query to the best adapter(s).

        Args:
            query: Input text to route.
            top_k: Override merge_config.top_k for this call.

        Returns:
            RoutingDecision with ranked selections and composer action.
        """
        k = top_k or self.merge_config.top_k
        start = time.perf_counter()

        selections = self.strategy.route(query, self.registry, top_k=k)
        action = self.composer.decide(selections)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Filter selections based on composer action
        if action.value == "single" and selections:
            active_selections = selections[:1]
        elif action.value == "fallback":
            active_selections = []
        else:
            active_selections = selections[:k]

        return RoutingDecision(
            selections=active_selections,
            action=action,
            strategy_used=self.strategy.name,
            composition_method=self.merge_config.method.value if action.value == "compose" else "",
            latency_ms=elapsed_ms,
            query=query,
        )

    def route_batch(
        self, queries: list[str], top_k: int | None = None
    ) -> list[RoutingDecision]:
        """Route a batch of queries.

        Args:
            queries: List of input texts.
            top_k: Override for max adapters per query.

        Returns:
            List of RoutingDecisions, one per query.
        """
        k = top_k or self.merge_config.top_k
        start = time.perf_counter()

        batch_selections = self.strategy.route_batch(
            queries, self.registry, top_k=k
        )

        total_ms = (time.perf_counter() - start) * 1000
        per_query_ms = total_ms / len(queries) if queries else 0.0

        results = []
        for query, selections in zip(queries, batch_selections):
            action = self.composer.decide(selections)

            if action.value == "single" and selections:
                active_selections = selections[:1]
            elif action.value == "fallback":
                active_selections = []
            else:
                active_selections = selections[:k]

            results.append(
                RoutingDecision(
                    selections=active_selections,
                    action=action,
                    strategy_used=self.strategy.name,
                    composition_method=(
                        self.merge_config.method.value if action.value == "compose" else ""
                    ),
                    latency_ms=per_query_ms,
                    query=query,
                )
            )
        return results

    def swap_strategy(self, strategy: BaseStrategy) -> None:
        """Hot-swap the routing strategy."""
        self.strategy = strategy

    def update_config(self, config: MergeConfig) -> None:
        """Update merge configuration and rebuild composer."""
        self.merge_config = config
        self.composer = SmartComposer(config)
