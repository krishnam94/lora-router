"""Unit tests for lora_router.strategies.ensemble module."""

from __future__ import annotations

import numpy as np
import pytest

from lora_router.strategies.ensemble import EnsembleStrategy
from lora_router.types import AdapterSelection

pytestmark = pytest.mark.unit


class _DummyStrategy:
    """Minimal strategy returning fixed selections for testing."""

    def __init__(self, name: str, selections: list[AdapterSelection]) -> None:
        self._name = name
        self._selections = selections

    @property
    def name(self) -> str:
        return self._name

    def route(self, query, registry, top_k=5):
        return self._selections[:top_k]

    def route_batch(self, queries, registry, top_k=5):
        return [self.route(q, registry, top_k) for q in queries]


class TestEnsembleStrategy:
    def test_empty_strategies_raises(self):
        """Initializing with empty strategies list raises ValueError."""
        with pytest.raises(ValueError, match="at least one sub-strategy"):
            EnsembleStrategy(strategies=[])

    def test_combines_multiple_strategies(self, mock_embedder, mock_registry):
        """Ensemble merges results from multiple sub-strategies."""
        s1 = _DummyStrategy(
            "s1",
            [
                AdapterSelection(adapter_name="code", confidence=0.8, scores={"s1": 0.8}),
                AdapterSelection(adapter_name="math", confidence=0.2, scores={"s1": 0.2}),
            ],
        )
        s2 = _DummyStrategy(
            "s2",
            [
                AdapterSelection(adapter_name="math", confidence=0.7, scores={"s2": 0.7}),
                AdapterSelection(adapter_name="code", confidence=0.3, scores={"s2": 0.3}),
            ],
        )
        ensemble = EnsembleStrategy(strategies=[(s1, 1.0), (s2, 1.0)])
        selections = ensemble.route("test query", mock_registry, top_k=5)
        assert len(selections) > 0
        adapter_names = {s.adapter_name for s in selections}
        assert "code" in adapter_names
        assert "math" in adapter_names

    def test_weighted_avg_aggregation(self, mock_embedder, mock_registry):
        """weighted_avg mode computes weighted average of confidences."""
        s1 = _DummyStrategy(
            "s1",
            [AdapterSelection(adapter_name="code", confidence=0.9, scores={"s1": 0.9})],
        )
        s2 = _DummyStrategy(
            "s2",
            [AdapterSelection(adapter_name="code", confidence=0.1, scores={"s2": 0.1})],
        )
        # Equal weights: avg = (0.9*0.5 + 0.1*0.5) / (0.5+0.5) = 0.5
        ensemble = EnsembleStrategy(
            strategies=[(s1, 1.0), (s2, 1.0)], aggregation="weighted_avg"
        )
        selections = ensemble.route("test", mock_registry, top_k=5)
        assert len(selections) == 1
        # After renormalization to sum=1, single adapter should have confidence=1.0
        np.testing.assert_almost_equal(selections[0].confidence, 1.0, decimal=3)

    def test_max_aggregation(self, mock_embedder, mock_registry):
        """max mode takes the maximum confidence per adapter."""
        s1 = _DummyStrategy(
            "s1",
            [
                AdapterSelection(adapter_name="code", confidence=0.9, scores={"s1": 0.9}),
                AdapterSelection(adapter_name="math", confidence=0.1, scores={"s1": 0.1}),
            ],
        )
        s2 = _DummyStrategy(
            "s2",
            [
                AdapterSelection(adapter_name="code", confidence=0.3, scores={"s2": 0.3}),
                AdapterSelection(adapter_name="math", confidence=0.7, scores={"s2": 0.7}),
            ],
        )
        ensemble = EnsembleStrategy(
            strategies=[(s1, 1.0), (s2, 1.0)], aggregation="max"
        )
        selections = ensemble.route("test", mock_registry, top_k=5)
        # code max = 0.9, math max = 0.7, so code should be first
        assert selections[0].adapter_name == "code"

    def test_scores_from_sub_strategies_preserved(self, mock_embedder, mock_registry):
        """Ensemble preserves per-strategy scores in the output."""
        s1 = _DummyStrategy(
            "s1",
            [AdapterSelection(adapter_name="code", confidence=0.8, scores={"s1_sim": 0.8})],
        )
        s2 = _DummyStrategy(
            "s2",
            [AdapterSelection(adapter_name="code", confidence=0.6, scores={"s2_seqr": 0.6})],
        )
        ensemble = EnsembleStrategy(strategies=[(s1, 1.0), (s2, 1.0)])
        selections = ensemble.route("test", mock_registry, top_k=5)
        assert len(selections) == 1
        assert "s1_sim" in selections[0].scores
        assert "s2_seqr" in selections[0].scores

    def test_name_property(self):
        """name property includes sub-strategy names."""
        s1 = _DummyStrategy("Sim", [])
        s2 = _DummyStrategy("SEQR", [])
        ensemble = EnsembleStrategy(strategies=[(s1, 1.0), (s2, 1.0)])
        assert "Sim" in ensemble.name
        assert "SEQR" in ensemble.name
        assert "Ensemble" in ensemble.name

    def test_confidences_sum_to_one(self, mock_embedder, mock_registry):
        """After renormalization, ensemble confidences sum to 1.0."""
        s1 = _DummyStrategy(
            "s1",
            [
                AdapterSelection(adapter_name="code", confidence=0.6, scores={"s1": 0.6}),
                AdapterSelection(adapter_name="math", confidence=0.4, scores={"s1": 0.4}),
            ],
        )
        ensemble = EnsembleStrategy(strategies=[(s1, 1.0)])
        selections = ensemble.route("test", mock_registry, top_k=5)
        total = sum(s.confidence for s in selections)
        np.testing.assert_almost_equal(total, 1.0, decimal=3)
