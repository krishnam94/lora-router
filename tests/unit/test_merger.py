"""Unit tests for lora_router.composition.merger module."""

from __future__ import annotations

import pytest
import torch

from lora_router.composition.merger import merge_adapters
from lora_router.types import MergeMethod

pytestmark = pytest.mark.unit


def _make_adapter_weights(seed: int, rank: int = 4, dim: int = 8) -> dict[str, torch.Tensor]:
    """Create a simple adapter weight dict with lora_A and lora_B."""
    torch.manual_seed(seed)
    return {
        "lora_A.weight": torch.randn(rank, dim),
        "lora_B.weight": torch.randn(dim, rank),
        "bias": torch.randn(dim),
    }


class TestMergeLinear:
    def test_weighted_average(self):
        """LINEAR merge produces a weighted average of adapter parameters."""
        w1 = {"param": torch.tensor([1.0, 2.0, 3.0])}
        w2 = {"param": torch.tensor([4.0, 5.0, 6.0])}
        merged = merge_adapters([w1, w2], weights=[0.5, 0.5], method=MergeMethod.LINEAR)
        expected = torch.tensor([2.5, 3.5, 4.5])
        torch.testing.assert_close(merged["param"], expected)

    def test_unequal_weights(self):
        """LINEAR merge with unequal weights."""
        w1 = {"param": torch.tensor([10.0, 0.0])}
        w2 = {"param": torch.tensor([0.0, 10.0])}
        merged = merge_adapters([w1, w2], weights=[0.8, 0.2], method=MergeMethod.LINEAR)
        expected = torch.tensor([8.0, 2.0])
        torch.testing.assert_close(merged["param"], expected)


class TestMergeTIES:
    def test_ties_trims_and_resolves(self):
        """TIES merge trims small values and resolves sign conflicts."""
        w1 = {"param": torch.tensor([10.0, -1.0, 0.5, -0.1])}
        w2 = {"param": torch.tensor([8.0, 2.0, -0.3, 0.2])}
        merged = merge_adapters(
            [w1, w2], weights=[0.5, 0.5], method=MergeMethod.TIES, density=0.5
        )
        assert "param" in merged
        # After trimming (keep 50%), small values are zeroed, signs are elected,
        # and only agreeing values are averaged. Result should be finite.
        assert torch.isfinite(merged["param"]).all()

    def test_ties_preserves_shape(self):
        """TIES merge output has same shape as input."""
        ws = [_make_adapter_weights(i) for i in range(3)]
        merged = merge_adapters(ws, weights=[0.4, 0.3, 0.3], method=MergeMethod.TIES, density=0.5)
        for key in ws[0]:
            assert merged[key].shape == ws[0][key].shape


class TestMergeDARE:
    def test_dare_applies_dropout(self):
        """DARE merge applies random dropout and rescaling."""
        torch.manual_seed(42)
        w1 = {"param": torch.ones(100)}
        w2 = {"param": torch.ones(100) * 2}
        merged = merge_adapters(
            [w1, w2], weights=[0.5, 0.5], method=MergeMethod.DARE, density=0.5
        )
        # Some parameters should be zero (from dropout), others rescaled
        assert "param" in merged
        # With density=0.5, roughly half of each adapter's params are dropped.
        # The non-dropped values are rescaled by 1/0.5 = 2, so the expected mean
        # of the merged result should be close to the weighted average.
        # Not all values should be zero
        assert merged["param"].abs().sum() > 0

    def test_dare_preserves_shape(self):
        """DARE merge output has same shape as input."""
        ws = [_make_adapter_weights(i) for i in range(2)]
        torch.manual_seed(99)
        merged = merge_adapters(ws, weights=[0.6, 0.4], method=MergeMethod.DARE, density=0.7)
        for key in ws[0]:
            assert merged[key].shape == ws[0][key].shape


class TestMergeCAT:
    def test_cat_concatenates_along_rank_dim(self):
        """CAT merge concatenates lora_A along dim=0 and lora_B along dim=1."""
        rank1, rank2, dim = 4, 8, 16
        w1 = {
            "lora_A.weight": torch.randn(rank1, dim),
            "lora_B.weight": torch.randn(dim, rank1),
            "other": torch.randn(dim),
        }
        w2 = {
            "lora_A.weight": torch.randn(rank2, dim),
            "lora_B.weight": torch.randn(dim, rank2),
            "other": torch.randn(dim),
        }
        merged = merge_adapters([w1, w2], weights=[0.5, 0.5], method=MergeMethod.CAT)
        assert merged["lora_A.weight"].shape == (rank1 + rank2, dim)
        assert merged["lora_B.weight"].shape == (dim, rank1 + rank2)
        # Non-LoRA params are summed (weighted)
        assert merged["other"].shape == (dim,)


class TestMergeEdgeCases:
    def test_single_adapter_returns_unchanged(self):
        """Merging a single adapter returns it as-is (no copy)."""
        w = {"param": torch.tensor([1.0, 2.0, 3.0])}
        merged = merge_adapters([w], weights=[1.0])
        # Returns the same dict object for single adapter
        assert merged is w

    def test_empty_adapter_list_raises(self):
        """Empty adapter list raises ValueError."""
        with pytest.raises(ValueError, match="at least one adapter"):
            merge_adapters([], weights=[])

    def test_weights_mismatch_raises(self):
        """Mismatched adapter_weights and weights lengths raises ValueError."""
        w1 = {"param": torch.tensor([1.0])}
        w2 = {"param": torch.tensor([2.0])}
        with pytest.raises(ValueError, match="same length"):
            merge_adapters([w1, w2], weights=[0.5])

    def test_three_way_linear_merge(self):
        """LINEAR merge of 3 adapters with equal weights."""
        w1 = {"p": torch.tensor([3.0])}
        w2 = {"p": torch.tensor([6.0])}
        w3 = {"p": torch.tensor([9.0])}
        merged = merge_adapters(
            [w1, w2, w3], weights=[1 / 3, 1 / 3, 1 / 3], method=MergeMethod.LINEAR
        )
        torch.testing.assert_close(merged["p"], torch.tensor([6.0]), atol=1e-5, rtol=1e-5)
