"""Unit tests for lora_router.types module."""

from __future__ import annotations

import pytest

from lora_router.types import (
    AdapterInfo,
    AdapterSelection,
    ComposerAction,
    MergeConfig,
    MergeMethod,
    RoutingDecision,
)

pytestmark = pytest.mark.unit


class TestAdapterInfo:
    def test_creation_minimal(self):
        """AdapterInfo with only required field (name) uses defaults."""
        info = AdapterInfo(name="test")
        assert info.name == "test"
        assert info.path == ""
        assert info.description == ""
        assert info.domain == ""
        assert info.example_queries == []
        assert info.metadata == {}

    def test_creation_full(self):
        """AdapterInfo with all fields populated."""
        info = AdapterInfo(
            name="code",
            path="/path/to/adapter",
            description="Code generation adapter",
            domain="code",
            example_queries=["Write a function", "Debug this"],
            metadata={"epochs": 3, "lr": 1e-4},
        )
        assert info.name == "code"
        assert info.path == "/path/to/adapter"
        assert info.domain == "code"
        assert len(info.example_queries) == 2
        assert info.metadata["epochs"] == 3

    def test_text_for_embedding_with_description_and_examples(self):
        """text_for_embedding joins description and up to 5 examples."""
        info = AdapterInfo(
            name="test",
            description="A test adapter",
            example_queries=["q1", "q2", "q3"],
        )
        text = info.text_for_embedding
        assert "A test adapter" in text
        assert "q1" in text
        assert "q2" in text
        assert "q3" in text

    def test_text_for_embedding_truncates_examples(self):
        """text_for_embedding only uses the first 5 examples."""
        queries = [f"query_{i}" for i in range(10)]
        info = AdapterInfo(name="test", description="desc", example_queries=queries)
        text = info.text_for_embedding
        assert "query_4" in text
        assert "query_5" not in text

    def test_text_for_embedding_falls_back_to_name(self):
        """text_for_embedding returns name when no description or examples."""
        info = AdapterInfo(name="fallback-adapter")
        assert info.text_for_embedding == "fallback-adapter"

    def test_has_examples_true(self):
        info = AdapterInfo(name="test", example_queries=["q1"])
        assert info.has_examples is True

    def test_has_examples_false(self):
        info = AdapterInfo(name="test")
        assert info.has_examples is False


class TestAdapterSelection:
    def test_valid_confidence(self):
        sel = AdapterSelection(adapter_name="code", confidence=0.85)
        assert sel.confidence == 0.85
        assert sel.scores == {}

    def test_confidence_bounds_zero(self):
        sel = AdapterSelection(adapter_name="code", confidence=0.0)
        assert sel.confidence == 0.0

    def test_confidence_bounds_one(self):
        sel = AdapterSelection(adapter_name="code", confidence=1.0)
        assert sel.confidence == 1.0

    def test_confidence_too_high_raises(self):
        with pytest.raises(ValueError):
            AdapterSelection(adapter_name="code", confidence=1.5)

    def test_confidence_negative_raises(self):
        with pytest.raises(ValueError):
            AdapterSelection(adapter_name="code", confidence=-0.1)

    def test_scores_dict(self):
        sel = AdapterSelection(
            adapter_name="code",
            confidence=0.9,
            scores={"similarity": 0.85, "seqr": 0.72},
        )
        assert sel.scores["similarity"] == 0.85
        assert sel.scores["seqr"] == 0.72


class TestRoutingDecision:
    def test_top_adapter(self):
        decision = RoutingDecision(
            selections=[
                AdapterSelection(adapter_name="code", confidence=0.9),
                AdapterSelection(adapter_name="math", confidence=0.1),
            ]
        )
        assert decision.top_adapter == "code"

    def test_top_confidence(self):
        decision = RoutingDecision(
            selections=[
                AdapterSelection(adapter_name="code", confidence=0.9),
            ]
        )
        assert decision.top_confidence == 0.9

    def test_adapter_names(self):
        decision = RoutingDecision(
            selections=[
                AdapterSelection(adapter_name="code", confidence=0.6),
                AdapterSelection(adapter_name="math", confidence=0.3),
                AdapterSelection(adapter_name="creative", confidence=0.1),
            ]
        )
        assert decision.adapter_names == ["code", "math", "creative"]

    def test_empty_selections(self):
        decision = RoutingDecision(selections=[])
        assert decision.top_adapter is None
        assert decision.top_confidence == 0.0
        assert decision.adapter_names == []

    def test_defaults(self):
        decision = RoutingDecision(
            selections=[AdapterSelection(adapter_name="x", confidence=0.5)]
        )
        assert decision.action == ComposerAction.SINGLE
        assert decision.strategy_used == ""
        assert decision.composition_method == ""
        assert decision.latency_ms == 0.0
        assert decision.query == ""


class TestMergeConfig:
    def test_defaults(self):
        config = MergeConfig()
        assert config.method == MergeMethod.LINEAR
        assert config.top_k == 3
        assert config.threshold_high == 0.8
        assert config.threshold_low == 0.3
        assert config.density == 0.5

    def test_custom_values(self):
        config = MergeConfig(
            method=MergeMethod.TIES,
            top_k=5,
            threshold_high=0.9,
            threshold_low=0.2,
            density=0.7,
        )
        assert config.method == MergeMethod.TIES
        assert config.top_k == 5
        assert config.threshold_high == 0.9
        assert config.threshold_low == 0.2
        assert config.density == 0.7


class TestEnums:
    def test_composer_action_values(self):
        assert ComposerAction.SINGLE.value == "single"
        assert ComposerAction.COMPOSE.value == "compose"
        assert ComposerAction.FALLBACK.value == "fallback"

    def test_merge_method_values(self):
        assert MergeMethod.LINEAR.value == "linear"
        assert MergeMethod.TIES.value == "ties"
        assert MergeMethod.DARE.value == "dare"
        assert MergeMethod.CAT.value == "cat"

    def test_enums_are_strings(self):
        """Both enums inherit from str, so they can be compared to strings."""
        assert ComposerAction.SINGLE == "single"
        assert MergeMethod.LINEAR == "linear"
