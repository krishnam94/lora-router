"""Ensemble routing strategy combining multiple sub-strategies.

Merges ranked adapter selections from multiple strategies using configurable
weights. This is lora-router's key differentiator - no competitor combines
semantic and spectral routing signals.
"""

from __future__ import annotations

from collections import defaultdict

from lora_router.registry import AdapterRegistry
from lora_router.strategies.base import BaseStrategy
from lora_router.types import AdapterSelection


class EnsembleStrategy(BaseStrategy):
    """Combine multiple routing strategies with weighted voting.

    Merges confidence scores from multiple strategies. Each strategy's scores are
    weighted by the strategy weight, then aggregated per adapter. The final
    confidence is a weighted average across all strategies that selected an adapter.

    Args:
        strategies: List of (strategy, weight) tuples.
        aggregation: How to combine scores - "weighted_avg" or "max".
    """

    def __init__(
        self,
        strategies: list[tuple[BaseStrategy, float]],
        aggregation: str = "weighted_avg",
    ) -> None:
        if not strategies:
            raise ValueError("EnsembleStrategy requires at least one sub-strategy")
        self._strategies = strategies
        self._aggregation = aggregation
        total_weight = sum(w for _, w in strategies)
        self._normalized_weights = [
            (s, w / total_weight) for s, w in strategies
        ]

    @property
    def name(self) -> str:
        sub_names = [s.name for s, _ in self._strategies]
        return f"Ensemble({'+'.join(sub_names)})"

    def route(
        self, query: str, registry: AdapterRegistry, top_k: int = 5
    ) -> list[AdapterSelection]:
        # Collect selections from all sub-strategies
        adapter_scores: dict[str, dict[str, float]] = defaultdict(dict)
        adapter_weighted_sum: dict[str, float] = defaultdict(float)
        adapter_weight_total: dict[str, float] = defaultdict(float)

        for strategy, weight in self._normalized_weights:
            selections = strategy.route(query, registry, top_k=top_k * 2)
            for sel in selections:
                adapter_scores[sel.adapter_name].update(sel.scores)
                if self._aggregation == "weighted_avg":
                    adapter_weighted_sum[sel.adapter_name] += sel.confidence * weight
                    adapter_weight_total[sel.adapter_name] += weight
                elif self._aggregation == "max":
                    current = adapter_weighted_sum.get(sel.adapter_name, 0.0)
                    adapter_weighted_sum[sel.adapter_name] = max(current, sel.confidence)

        # Compute final confidence per adapter
        final_scores: list[tuple[str, float, dict[str, float]]] = []
        for adapter_name, weighted_sum in adapter_weighted_sum.items():
            if self._aggregation == "weighted_avg":
                weight_total = adapter_weight_total[adapter_name]
                confidence = weighted_sum / weight_total if weight_total > 0 else 0.0
            else:
                confidence = weighted_sum
            final_scores.append((adapter_name, confidence, adapter_scores[adapter_name]))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        final_scores = final_scores[:top_k]

        # Re-normalize confidences to sum to 1
        if final_scores:
            total_conf = sum(c for _, c, _ in final_scores)
            if total_conf > 0:
                final_scores = [
                    (name, conf / total_conf, scores)
                    for name, conf, scores in final_scores
                ]

        return [
            AdapterSelection(
                adapter_name=name,
                confidence=float(conf),
                scores=scores,
            )
            for name, conf, scores in final_scores
        ]

    def route_batch(
        self, queries: list[str], registry: AdapterRegistry, top_k: int = 5
    ) -> list[list[AdapterSelection]]:
        return [self.route(q, registry, top_k=top_k) for q in queries]
