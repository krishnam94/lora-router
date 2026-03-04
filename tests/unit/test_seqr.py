"""Unit tests for lora_router.strategies.seqr module."""

from __future__ import annotations

import numpy as np
import pytest

from lora_router.registry import AdapterRegistry
from lora_router.strategies.seqr import SEQRStrategy
from lora_router.types import AdapterInfo

pytestmark = pytest.mark.unit


class TestSEQRStrategy:
    def test_route_with_signatures(
        self, mock_embedder, mock_registry, sample_seqr_signatures
    ):
        """route() returns selections when signatures are set."""
        strategy = SEQRStrategy(temperature=0.2, use_qr=True)
        strategy.set_signatures(sample_seqr_signatures)
        selections = strategy.route("Write Python code", mock_registry, top_k=3)
        assert len(selections) > 0
        assert len(selections) <= 3
        # Check descending confidence
        confidences = [s.confidence for s in selections]
        assert confidences == sorted(confidences, reverse=True)

    def test_empty_signatures_returns_empty(self, mock_embedder, mock_registry):
        """route() returns empty list when no signatures are set."""
        strategy = SEQRStrategy()
        selections = strategy.route("Write code", mock_registry)
        assert selections == []

    def test_no_embedder_returns_empty(self, sample_seqr_signatures):
        """route() returns empty when registry has no embedder."""
        registry = AdapterRegistry(embedder=None)
        registry.register(AdapterInfo(name="code"), compute_embedding=False)
        strategy = SEQRStrategy()
        strategy.set_signatures(sample_seqr_signatures)
        selections = strategy.route("Write code", registry)
        assert selections == []

    def test_compute_signature_from_weights_qr(self):
        """compute_signature_from_weights with QR returns correct shape."""
        rng = np.random.RandomState(123)
        rank = 8
        in_features = 64
        out_features = 128
        lora_A = rng.randn(rank, in_features).astype(np.float32)
        lora_B = rng.randn(out_features, rank).astype(np.float32)
        sig = SEQRStrategy.compute_signature_from_weights(lora_A, lora_B, use_qr=True)
        # product = lora_B @ lora_A => (128, 64)
        # product.T => (64, 128)
        # np.linalg.qr((64, 128), mode='reduced'): Q=(64,64), R=(64,128)
        min_dim = min(in_features, out_features)
        assert sig.ndim == 2
        assert sig.shape == (min_dim, out_features)

    def test_compute_signature_from_weights_svd(self):
        """compute_signature_from_weights with SVD returns correct shape."""
        rng = np.random.RandomState(456)
        rank = 4
        in_features = 32
        out_features = 64
        lora_A = rng.randn(rank, in_features).astype(np.float32)
        lora_B = rng.randn(out_features, rank).astype(np.float32)
        sig = SEQRStrategy.compute_signature_from_weights(lora_A, lora_B, use_qr=False)
        # SVD on product (out_features, in_features) with full_matrices=False:
        # product shape = (64, 32)
        # U: (64, 32), S: (32,), Vt: (32, 32)
        # Returns diag(S) @ Vt => (32, 32) = (min(out, in), in_features)
        min_dim = min(out_features, in_features)
        assert sig.ndim == 2
        assert sig.shape == (min_dim, in_features)

    def test_activation_norm_computation(self, sample_seqr_signatures):
        """_compute_activation_norm returns a positive scalar."""
        strategy = SEQRStrategy()
        query_vec = np.random.randn(64).astype(np.float32)
        sig = sample_seqr_signatures["code"]
        norm = strategy._compute_activation_norm(query_vec, sig)
        assert isinstance(norm, float)
        assert norm >= 0.0

    def test_scores_contain_seqr_key(
        self, mock_embedder, mock_registry, sample_seqr_signatures
    ):
        """Each selection should have 'seqr' in its scores dict."""
        strategy = SEQRStrategy()
        strategy.set_signatures(sample_seqr_signatures)
        selections = strategy.route("Solve this math problem", mock_registry, top_k=6)
        for sel in selections:
            assert "seqr" in sel.scores
            assert isinstance(sel.scores["seqr"], float)

    def test_confidences_sum_to_roughly_one(
        self, mock_embedder, mock_registry, sample_seqr_signatures
    ):
        """Softmax confidences should sum to approximately 1.0."""
        strategy = SEQRStrategy()
        strategy.set_signatures(sample_seqr_signatures)
        selections = strategy.route("Translate this", mock_registry, top_k=6)
        total = sum(s.confidence for s in selections)
        np.testing.assert_almost_equal(total, 1.0, decimal=3)
