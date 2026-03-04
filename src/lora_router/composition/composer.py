"""Smart composer for confidence-based adapter selection/composition/fallback."""

from __future__ import annotations

from lora_router.types import AdapterSelection, ComposerAction, MergeConfig


class SmartComposer:
    """Decides whether to use a single adapter, compose multiple, or fall back.

    The key insight: not every query needs adapter composition. High-confidence
    routing should use a single adapter (fast path). Low-confidence routing
    should fall back to the base model. Composition is for the middle ground
    where the router is uncertain which single adapter is best.

    This is a differentiator vs LORAUTER (always K=3) and LoraRetriever (always top-k).

    Args:
        config: Merge configuration with confidence thresholds.
    """

    def __init__(self, config: MergeConfig | None = None) -> None:
        self.config = config or MergeConfig()

    def decide(self, selections: list[AdapterSelection]) -> ComposerAction:
        """Decide the composition action based on routing confidence.

        Decision logic:
        - If no selections: FALLBACK
        - If top confidence > threshold_high: SINGLE (use best adapter)
        - If top confidence < threshold_low: FALLBACK (use base model)
        - Otherwise: COMPOSE (merge top-k adapters)

        Args:
            selections: Ranked adapter selections from a strategy.

        Returns:
            ComposerAction indicating what to do.
        """
        if not selections:
            return ComposerAction.FALLBACK

        top_confidence = selections[0].confidence

        if top_confidence >= self.config.threshold_high:
            return ComposerAction.SINGLE
        elif top_confidence <= self.config.threshold_low:
            return ComposerAction.FALLBACK
        else:
            return ComposerAction.COMPOSE

    def get_merge_weights(self, selections: list[AdapterSelection]) -> list[float]:
        """Compute merge weights proportional to confidence scores.

        Unlike LoraRetriever (uniform 1/k) or LORAUTER (softmax with temperature),
        we use confidence-proportional weights from the routing strategy.
        This means more relevant adapters contribute more to the composition.

        Args:
            selections: Adapter selections to compose.

        Returns:
            List of weights summing to 1.0.
        """
        if not selections:
            return []

        k = min(self.config.top_k, len(selections))
        top_selections = selections[:k]

        confidences = [s.confidence for s in top_selections]
        total = sum(confidences)

        if total == 0:
            return [1.0 / k] * k

        return [c / total for c in confidences]
