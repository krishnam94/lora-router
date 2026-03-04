"""Adapter merging utilities.

Supports multiple merge strategies:
- LINEAR: Weighted average of adapter parameters
- TIES: TrIm, Elect Sign, and merge - resolves sign conflicts (arXiv:2306.01708)
- DARE: Drop And REscale - prunes redundant parameters (arXiv:2311.03099)
- CAT: Concatenation - stacks adapter matrices (LoRA Soups, arXiv:2410.13025)
"""

from __future__ import annotations

import torch

from lora_router.types import MergeMethod


def merge_adapters(
    adapter_weights: list[dict[str, torch.Tensor]],
    weights: list[float],
    method: MergeMethod = MergeMethod.LINEAR,
    density: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Merge multiple adapter weight dicts into one.

    Args:
        adapter_weights: List of state dicts (key -> tensor) for each adapter.
        weights: Scalar weight per adapter (should sum to 1 for LINEAR).
        method: Merge strategy to use.
        density: For TIES/DARE, fraction of parameters to keep.

    Returns:
        Merged state dict.
    """
    if not adapter_weights:
        raise ValueError("Need at least one adapter to merge")
    if len(adapter_weights) != len(weights):
        raise ValueError("adapter_weights and weights must have same length")
    if len(adapter_weights) == 1:
        return adapter_weights[0]

    if method == MergeMethod.LINEAR:
        return _merge_linear(adapter_weights, weights)
    elif method == MergeMethod.TIES:
        return _merge_ties(adapter_weights, weights, density)
    elif method == MergeMethod.DARE:
        return _merge_dare(adapter_weights, weights, density)
    elif method == MergeMethod.CAT:
        return _merge_cat(adapter_weights, weights)
    else:
        raise ValueError(f"Unknown merge method: {method}")


def _merge_linear(
    adapter_weights: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Simple weighted average of parameters."""
    merged: dict[str, torch.Tensor] = {}
    keys = adapter_weights[0].keys()

    for key in keys:
        merged[key] = sum(
            w * aw[key].float() for w, aw in zip(weights, adapter_weights)
        )  # type: ignore[assignment]

    return merged


def _merge_ties(
    adapter_weights: list[dict[str, torch.Tensor]],
    weights: list[float],
    density: float,
) -> dict[str, torch.Tensor]:
    """TIES-Merging: Trim, Elect Sign, Merge.

    1. Trim: Keep only top-density% parameters by magnitude per adapter.
    2. Elect Sign: For each parameter position, use the sign that has the most
       total magnitude across adapters.
    3. Merge: Average only the parameters that agree with the elected sign.
    """
    merged: dict[str, torch.Tensor] = {}
    keys = adapter_weights[0].keys()

    for key in keys:
        tensors = [aw[key].float() for aw in adapter_weights]
        weighted = [t * w for t, w in zip(tensors, weights)]

        # Step 1: Trim - zero out small-magnitude parameters
        trimmed = []
        for t in weighted:
            flat = t.abs().flatten()
            if len(flat) == 0:
                trimmed.append(t)
                continue
            k = max(1, int(density * len(flat)))
            threshold = torch.topk(flat, k).values[-1]
            mask = t.abs() >= threshold
            trimmed.append(t * mask)

        # Step 2: Elect sign - majority vote by magnitude
        stacked = torch.stack(trimmed)
        sign_sum = torch.sum(stacked, dim=0)
        elected_sign = torch.sign(sign_sum)
        elected_sign[elected_sign == 0] = 1  # default positive

        # Step 3: Merge - average values that agree with elected sign
        agreement_sum = torch.zeros_like(trimmed[0])
        agreement_count = torch.zeros_like(trimmed[0])
        for t in trimmed:
            agrees = torch.sign(t) == elected_sign
            agreement_sum += t * agrees
            agreement_count += agrees.float()

        agreement_count = torch.clamp(agreement_count, min=1)
        merged[key] = agreement_sum / agreement_count

    return merged


def _merge_dare(
    adapter_weights: list[dict[str, torch.Tensor]],
    weights: list[float],
    density: float,
) -> dict[str, torch.Tensor]:
    """DARE: Drop And REscale.

    1. For each adapter, randomly drop (1-density)% of parameters.
    2. Rescale remaining by 1/density to maintain expected magnitude.
    3. Merge with weighted average.
    """
    merged: dict[str, torch.Tensor] = {}
    keys = adapter_weights[0].keys()

    for key in keys:
        tensors = [aw[key].float() for aw in adapter_weights]

        rescaled = []
        for t in tensors:
            mask = torch.bernoulli(torch.full_like(t, density))
            rescaled.append(t * mask / max(density, 1e-8))

        merged[key] = sum(
            w * r for w, r in zip(weights, rescaled)
        )  # type: ignore[assignment]

    return merged


def _merge_cat(
    adapter_weights: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """CAT: Concatenate adapter matrices along the rank dimension.

    For LoRA, concatenates A matrices along dim 0 (rank) and B matrices along dim 1.
    The result is a higher-rank adapter that captures all adapters' knowledge.
    Weights are used to scale each adapter's contribution.
    """
    merged: dict[str, torch.Tensor] = {}
    keys = adapter_weights[0].keys()

    for key in keys:
        tensors = [aw[key].float() * w for w, aw in zip(weights, adapter_weights)]

        if "lora_A" in key:
            # A matrices: (rank, in_features) -> concat along rank dim
            merged[key] = torch.cat(tensors, dim=0)
        elif "lora_B" in key:
            # B matrices: (out_features, rank) -> concat along rank dim
            merged[key] = torch.cat(tensors, dim=1)
        else:
            # Other parameters: weighted average
            merged[key] = sum(tensors)  # type: ignore[assignment]

    return merged
