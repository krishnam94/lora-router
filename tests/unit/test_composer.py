"""Unit tests for lora_router.composition.composer module."""

from __future__ import annotations

import numpy as np
import pytest

from lora_router.composition.composer import SmartComposer
from lora_router.types import AdapterSelection, ComposerAction, MergeConfig

pytestmark = pytest.mark.unit


class TestSmartComposer:
    def test_high_confidence_returns_single(self):
        """Top confidence above threshold_high produces SINGLE action."""
        config = MergeConfig(threshold_high=0.8, threshold_low=0.3)
        composer = SmartComposer(config)
        selections = [
            AdapterSelection(adapter_name="code", confidence=0.95),
            AdapterSelection(adapter_name="math", confidence=0.05),
        ]
        assert composer.decide(selections) == ComposerAction.SINGLE

    def test_low_confidence_returns_fallback(self):
        """Top confidence below threshold_low produces FALLBACK action."""
        config = MergeConfig(threshold_high=0.8, threshold_low=0.3)
        composer = SmartComposer(config)
        selections = [
            AdapterSelection(adapter_name="code", confidence=0.2),
            AdapterSelection(adapter_name="math", confidence=0.1),
        ]
        assert composer.decide(selections) == ComposerAction.FALLBACK

    def test_medium_confidence_returns_compose(self):
        """Top confidence between thresholds produces COMPOSE action."""
        config = MergeConfig(threshold_high=0.8, threshold_low=0.3)
        composer = SmartComposer(config)
        selections = [
            AdapterSelection(adapter_name="code", confidence=0.5),
            AdapterSelection(adapter_name="math", confidence=0.3),
            AdapterSelection(adapter_name="creative", confidence=0.2),
        ]
        assert composer.decide(selections) == ComposerAction.COMPOSE

    def test_empty_selections_returns_fallback(self):
        """Empty selection list produces FALLBACK action."""
        composer = SmartComposer()
        assert composer.decide([]) == ComposerAction.FALLBACK

    def test_get_merge_weights_sum_to_one(self):
        """get_merge_weights returns weights that sum to 1.0."""
        config = MergeConfig(top_k=3)
        composer = SmartComposer(config)
        selections = [
            AdapterSelection(adapter_name="code", confidence=0.6),
            AdapterSelection(adapter_name="math", confidence=0.3),
            AdapterSelection(adapter_name="creative", confidence=0.1),
        ]
        weights = composer.get_merge_weights(selections)
        assert len(weights) == 3
        np.testing.assert_almost_equal(sum(weights), 1.0, decimal=6)
        # Weights proportional to confidence
        assert weights[0] > weights[1] > weights[2]

    def test_get_merge_weights_respects_top_k(self):
        """get_merge_weights only returns top_k weights."""
        config = MergeConfig(top_k=2)
        composer = SmartComposer(config)
        selections = [
            AdapterSelection(adapter_name="code", confidence=0.5),
            AdapterSelection(adapter_name="math", confidence=0.3),
            AdapterSelection(adapter_name="creative", confidence=0.2),
        ]
        weights = composer.get_merge_weights(selections)
        assert len(weights) == 2
        np.testing.assert_almost_equal(sum(weights), 1.0, decimal=6)

    def test_get_merge_weights_empty_returns_empty(self):
        """get_merge_weights returns empty list for empty selections."""
        composer = SmartComposer()
        assert composer.get_merge_weights([]) == []

    def test_custom_thresholds(self):
        """Custom threshold values change decision boundaries."""
        config = MergeConfig(threshold_high=0.95, threshold_low=0.5)
        composer = SmartComposer(config)

        # 0.9 is below 0.95, above 0.5, so COMPOSE
        selections_compose = [
            AdapterSelection(adapter_name="code", confidence=0.9),
        ]
        assert composer.decide(selections_compose) == ComposerAction.COMPOSE

        # 0.96 is above 0.95, so SINGLE
        selections_single = [
            AdapterSelection(adapter_name="code", confidence=0.96),
        ]
        assert composer.decide(selections_single) == ComposerAction.SINGLE

        # 0.4 is below 0.5, so FALLBACK
        selections_fallback = [
            AdapterSelection(adapter_name="code", confidence=0.4),
        ]
        assert composer.decide(selections_fallback) == ComposerAction.FALLBACK

    def test_boundary_at_threshold_high(self):
        """Confidence exactly equal to threshold_high produces SINGLE."""
        config = MergeConfig(threshold_high=0.8, threshold_low=0.3)
        composer = SmartComposer(config)
        selections = [AdapterSelection(adapter_name="code", confidence=0.8)]
        assert composer.decide(selections) == ComposerAction.SINGLE

    def test_boundary_at_threshold_low(self):
        """Confidence exactly equal to threshold_low produces FALLBACK."""
        config = MergeConfig(threshold_high=0.8, threshold_low=0.3)
        composer = SmartComposer(config)
        selections = [AdapterSelection(adapter_name="code", confidence=0.3)]
        assert composer.decide(selections) == ComposerAction.FALLBACK

    def test_get_merge_weights_zero_confidence(self):
        """Zero-confidence selections get uniform weights."""
        config = MergeConfig(top_k=2)
        composer = SmartComposer(config)
        selections = [
            AdapterSelection(adapter_name="a", confidence=0.0),
            AdapterSelection(adapter_name="b", confidence=0.0),
        ]
        weights = composer.get_merge_weights(selections)
        assert len(weights) == 2
        np.testing.assert_almost_equal(weights[0], 0.5, decimal=6)
        np.testing.assert_almost_equal(weights[1], 0.5, decimal=6)
