"""Unit tests for lora_router.router module."""

from __future__ import annotations

import pytest

from lora_router.router import LoRARouter
from lora_router.strategies.similarity import SimilarityStrategy
from lora_router.types import ComposerAction, MergeConfig, RoutingDecision

pytestmark = pytest.mark.unit


class TestLoRARouter:
    def test_route_returns_routing_decision(self, mock_embedder, mock_registry):
        """route() returns a RoutingDecision object."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        router = LoRARouter(mock_registry, strategy)
        decision = router.route("Write Python code")
        assert isinstance(decision, RoutingDecision)
        assert decision.query == "Write Python code"

    def test_route_batch_returns_list(self, mock_embedder, mock_registry):
        """route_batch() returns one RoutingDecision per query."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        router = LoRARouter(mock_registry, strategy)
        queries = ["Write code", "Solve math", "Translate text"]
        decisions = router.route_batch(queries)
        assert len(decisions) == 3
        for d in decisions:
            assert isinstance(d, RoutingDecision)

    def test_latency_ms_is_recorded(self, mock_embedder, mock_registry):
        """latency_ms is a positive value after routing."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        router = LoRARouter(mock_registry, strategy)
        decision = router.route("Write Python code")
        assert decision.latency_ms >= 0.0

    def test_strategy_used_is_set(self, mock_embedder, mock_registry):
        """strategy_used field contains the strategy class name."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        router = LoRARouter(mock_registry, strategy)
        decision = router.route("Write code")
        assert decision.strategy_used == "SimilarityStrategy"

    def test_swap_strategy_changes_behavior(self, mock_embedder, mock_registry):
        """swap_strategy() changes the active strategy."""
        strategy1 = SimilarityStrategy(encoder=mock_embedder, temperature=0.1)
        strategy2 = SimilarityStrategy(encoder=mock_embedder, temperature=2.0)
        router = LoRARouter(mock_registry, strategy1)

        decision1 = router.route("Write code")
        router.swap_strategy(strategy2)
        decision2 = router.route("Write code")

        # After swap, the strategy_used name stays the same (both SimilarityStrategy),
        # but the confidence distribution should differ due to temperature change
        assert decision1.strategy_used == decision2.strategy_used
        # With temperature=0.1, confidences are more peaked
        # With temperature=2.0, confidences are more uniform
        # The top confidence with low temp should be higher
        if decision1.selections and decision2.selections:
            assert decision1.top_confidence != decision2.top_confidence

    def test_high_confidence_produces_single_action(self, mock_embedder, mock_registry):
        """Very peaked confidence (low temperature) should produce SINGLE action."""
        # Low temperature makes the top adapter get very high confidence
        strategy = SimilarityStrategy(encoder=mock_embedder, temperature=0.01)
        config = MergeConfig(threshold_high=0.8, threshold_low=0.3)
        router = LoRARouter(mock_registry, strategy, merge_config=config)
        decision = router.route("Write Python code")
        # With extremely low temperature, top confidence will be very close to 1.0
        assert decision.action == ComposerAction.SINGLE

    def test_update_config_rebuilds_composer(self, mock_embedder, mock_registry):
        """update_config() changes merge config and rebuilds the composer."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        initial_config = MergeConfig(top_k=3)
        router = LoRARouter(mock_registry, strategy, merge_config=initial_config)

        new_config = MergeConfig(top_k=5, threshold_high=0.95)
        router.update_config(new_config)
        assert router.merge_config.top_k == 5
        assert router.merge_config.threshold_high == 0.95
