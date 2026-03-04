"""Shared utility functions for routing strategies."""

from __future__ import annotations

import numpy as np


def softmax_confidence(scores: np.ndarray, temperature: float = 0.2) -> np.ndarray:
    """Convert raw scores to calibrated confidence via temperature-scaled softmax.

    Used by SimilarityStrategy and SEQRStrategy to produce comparable
    confidence values from raw similarity/activation scores.

    Args:
        scores: Raw scores (similarities, activation norms, etc.)
        temperature: Controls peakedness. Lower = more peaked. Default 0.2 matches LORAUTER.

    Returns:
        Confidence array summing to ~1.0.
    """
    scaled = scores / temperature
    scaled = scaled - scaled.max()
    exp_scores = np.exp(scaled)
    return exp_scores / (exp_scores.sum() + 1e-8)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of row vectors.

    Args:
        query: Shape (D,) query vector.
        matrix: Shape (N, D) matrix of candidate vectors.

    Returns:
        Shape (N,) similarity scores.
    """
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norms @ query_norm
