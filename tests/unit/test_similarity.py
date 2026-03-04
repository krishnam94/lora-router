"""Unit tests for lora_router.strategies.similarity module."""

from __future__ import annotations

import numpy as np
import pytest

from lora_router.registry import AdapterRegistry
from lora_router.strategies.similarity import SimilarityStrategy, _cosine_similarity
from lora_router.types import AdapterInfo

pytestmark = pytest.mark.unit


class TestCosineSimHelper:
    def test_identical_vectors(self):
        """Cosine similarity of identical vectors is 1.0."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sims = _cosine_similarity(a, b)
        assert sims.shape == (1,)
        np.testing.assert_almost_equal(sims[0], 1.0, decimal=5)

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors is 0.0."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        sims = _cosine_similarity(a, b)
        np.testing.assert_almost_equal(sims[0], 0.0, decimal=5)

    def test_opposite_vectors(self):
        """Cosine similarity of opposite vectors is -1.0."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        sims = _cosine_similarity(a, b)
        np.testing.assert_almost_equal(sims[0], -1.0, decimal=5)

    def test_multiple_rows(self):
        """Cosine similarity against a matrix of multiple rows."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.707, 0.707, 0.0],
            ],
            dtype=np.float32,
        )
        sims = _cosine_similarity(a, b)
        assert sims.shape == (3,)
        np.testing.assert_almost_equal(sims[0], 1.0, decimal=3)
        np.testing.assert_almost_equal(sims[1], 0.0, decimal=3)
        assert sims[2] > 0.5  # Partially aligned


class TestSimilarityStrategy:
    def test_route_returns_ranked_selections(self, mock_embedder, mock_registry):
        """route() returns AdapterSelection objects sorted by confidence descending."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        selections = strategy.route("Write Python code", mock_registry, top_k=6)
        assert len(selections) > 0
        assert len(selections) <= 6
        # Check descending confidence order
        confidences = [s.confidence for s in selections]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidence_scores_sum_to_one(self, mock_embedder, mock_registry):
        """Softmax-based confidences should approximately sum to 1."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        selections = strategy.route("Solve this math equation", mock_registry, top_k=6)
        total = sum(s.confidence for s in selections)
        np.testing.assert_almost_equal(total, 1.0, decimal=3)

    def test_top_k_limits_results(self, mock_embedder, mock_registry):
        """top_k parameter limits the number of returned selections."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        selections = strategy.route("Write code", mock_registry, top_k=2)
        assert len(selections) <= 2

    def test_empty_registry_returns_empty(self, mock_embedder):
        """Empty registry produces no selections."""
        empty_registry = AdapterRegistry(embedder=mock_embedder)
        strategy = SimilarityStrategy(encoder=mock_embedder)
        selections = strategy.route("anything", empty_registry)
        assert selections == []

    def test_route_batch_returns_correct_count(self, mock_embedder, mock_registry):
        """route_batch() returns one result list per query."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        queries = ["Write code", "Solve math", "Translate text"]
        results = strategy.route_batch(queries, mock_registry, top_k=3)
        assert len(results) == 3
        for result in results:
            assert len(result) <= 3

    def test_scores_dict_contains_similarity(self, mock_embedder, mock_registry):
        """Each AdapterSelection should have a 'similarity' key in scores."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        selections = strategy.route("Write code", mock_registry, top_k=3)
        for sel in selections:
            assert "similarity" in sel.scores
            assert isinstance(sel.scores["similarity"], float)

    def test_all_adapter_names_valid(self, mock_embedder, mock_registry):
        """All returned adapter names should exist in the registry."""
        strategy = SimilarityStrategy(encoder=mock_embedder)
        selections = strategy.route("Debug this function", mock_registry, top_k=6)
        valid_names = set(mock_registry.adapter_names)
        for sel in selections:
            assert sel.adapter_name in valid_names

    def test_single_adapter_returns_confidence_one(self, mock_embedder):
        """With only one adapter, confidence should be ~1.0."""
        registry = AdapterRegistry(embedder=mock_embedder)
        registry.register(AdapterInfo(name="solo", description="The only adapter"))
        strategy = SimilarityStrategy(encoder=mock_embedder)
        selections = strategy.route("anything", registry, top_k=5)
        assert len(selections) == 1
        np.testing.assert_almost_equal(selections[0].confidence, 1.0, decimal=3)
