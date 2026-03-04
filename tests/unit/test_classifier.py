"""Unit tests for lora_router.strategies.classifier module."""

from __future__ import annotations

import pytest

from lora_router.registry import AdapterRegistry
from lora_router.strategies.classifier import ClassifierStrategy
from lora_router.types import AdapterInfo

pytestmark = pytest.mark.unit


class TestClassifierStrategy:
    def test_route_before_training_returns_empty(self, mock_embedder, mock_registry):
        """route() returns empty list when classifier is not trained."""
        strategy = ClassifierStrategy(encoder=mock_embedder)
        selections = strategy.route("Write code", mock_registry)
        assert selections == []

    def test_train_raises_with_insufficient_data(self, mock_embedder):
        """train() raises ValueError when fewer than 2 adapters have examples."""
        registry = AdapterRegistry(embedder=mock_embedder)
        registry.register(
            AdapterInfo(name="only_one", example_queries=["query1", "query2"])
        )
        strategy = ClassifierStrategy(encoder=mock_embedder)
        with pytest.raises(ValueError, match="Need at least 2 adapters"):
            strategy.train(registry)

    def test_train_raises_no_examples(self, mock_embedder):
        """train() raises when no adapters have example queries."""
        registry = AdapterRegistry(embedder=mock_embedder)
        registry.register(AdapterInfo(name="a"))
        registry.register(AdapterInfo(name="b"))
        strategy = ClassifierStrategy(encoder=mock_embedder)
        with pytest.raises(ValueError, match="Need at least 2 adapters"):
            strategy.train(registry)

    def test_train_and_route(self, mock_embedder, mock_registry):
        """After training, route() returns valid selections."""
        strategy = ClassifierStrategy(encoder=mock_embedder, calibrated=False)
        strategy.train(mock_registry)
        selections = strategy.route("Write a Python function", mock_registry, top_k=3)
        assert len(selections) > 0
        assert len(selections) <= 3
        for sel in selections:
            assert 0.0 <= sel.confidence <= 1.0

    def test_confidence_scores_are_probabilities(self, mock_embedder, mock_registry):
        """Classifier confidences should be valid probabilities."""
        strategy = ClassifierStrategy(encoder=mock_embedder, calibrated=False)
        strategy.train(mock_registry)
        selections = strategy.route("Solve math equation", mock_registry, top_k=6)
        # All confidences should be between 0 and 1
        for sel in selections:
            assert 0.0 <= sel.confidence <= 1.0
        # Probabilities from predict_proba across all classes sum to 1,
        # but we may return top_k subset. Each is still a valid probability.

    def test_scores_contain_classifier_key(self, mock_embedder, mock_registry):
        """Each selection should have 'classifier' in scores dict."""
        strategy = ClassifierStrategy(encoder=mock_embedder, calibrated=False)
        strategy.train(mock_registry)
        selections = strategy.route("Translate text", mock_registry, top_k=3)
        for sel in selections:
            assert "classifier" in sel.scores
            assert isinstance(sel.scores["classifier"], float)

    def test_is_trained_flag(self, mock_embedder, mock_registry):
        """_is_trained flag is False before training and True after."""
        strategy = ClassifierStrategy(encoder=mock_embedder, calibrated=False)
        assert strategy._is_trained is False
        strategy.train(mock_registry)
        assert strategy._is_trained is True
