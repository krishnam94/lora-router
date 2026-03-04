"""Evaluation metrics for routing quality.

Implements the standard metrics used in LoRA routing papers:
- Normalized oracle score (LORAUTER's primary metric)
- Routing accuracy (top-1, top-k)
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Per-cluster breakdown
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def routing_accuracy(
    predictions: list[str],
    ground_truth: list[str],
) -> float:
    """Top-1 routing accuracy: fraction of queries routed to the correct adapter.

    Args:
        predictions: Predicted adapter names (one per query).
        ground_truth: Ground-truth adapter names.

    Returns:
        Accuracy between 0.0 and 1.0.
    """
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def routing_accuracy_at_k(
    predictions: list[list[str]],
    ground_truth: list[str],
    k: int = 3,
) -> float:
    """Top-k routing accuracy: fraction where correct adapter is in top-k.

    Args:
        predictions: Ranked lists of adapter names per query.
        ground_truth: Ground-truth adapter names.
        k: Consider top-k predictions.

    Returns:
        Accuracy between 0.0 and 1.0.
    """
    if not predictions:
        return 0.0
    correct = sum(g in preds[:k] for preds, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def mean_reciprocal_rank(
    predictions: list[list[str]],
    ground_truth: list[str],
) -> float:
    """Mean Reciprocal Rank (MRR).

    For each query, finds the rank of the correct adapter in the predicted list.
    MRR = mean(1/rank) across queries. Higher is better.

    Args:
        predictions: Ranked lists of adapter names per query.
        ground_truth: Ground-truth adapter names.

    Returns:
        MRR between 0.0 and 1.0.
    """
    if not predictions:
        return 0.0
    reciprocal_ranks = []
    for preds, gt in zip(predictions, ground_truth):
        try:
            rank = preds.index(gt) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks))


def ndcg(
    predictions: list[list[str]],
    ground_truth: list[str],
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Binary relevance: 1 if adapter matches ground truth, 0 otherwise.

    Args:
        predictions: Ranked lists of adapter names per query.
        ground_truth: Ground-truth adapter names.
        k: Truncation point.

    Returns:
        NDCG@k between 0.0 and 1.0.
    """
    if not predictions:
        return 0.0

    ndcg_scores = []
    for preds, gt in zip(predictions, ground_truth):
        # DCG: relevance / log2(rank + 1) for each position
        dcg = 0.0
        for i, pred in enumerate(preds[:k]):
            if pred == gt:
                dcg += 1.0 / np.log2(i + 2)  # +2 because i starts at 0

        # Ideal DCG (ground truth at rank 1)
        idcg = 1.0 / np.log2(2)  # 1/log2(2) = 1.0

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores))


def normalized_oracle_score(
    method_scores: dict[str, float],
    oracle_scores: dict[str, float],
) -> float:
    """Normalized oracle score - LORAUTER's primary metric.

    For each task, computes method_score / oracle_score, then averages.
    100% means matching the oracle (perfect per-task adapter selection).
    >100% is possible when composition outperforms any single adapter.

    Args:
        method_scores: Task name -> method's score on that task.
        oracle_scores: Task name -> oracle (best single adapter) score.

    Returns:
        Normalized score as a percentage (e.g., 88.4 means 88.4% of oracle).
    """
    if not method_scores or not oracle_scores:
        return 0.0

    normalized = []
    for task, method_score in method_scores.items():
        oracle_score = oracle_scores.get(task, 0.0)
        if oracle_score > 0:
            normalized.append(method_score / oracle_score * 100)

    return float(np.mean(normalized)) if normalized else 0.0


def per_cluster_scores(
    task_scores: dict[str, float],
    task_clusters: dict[str, str],
    oracle_scores: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-cluster average scores and normalized scores.

    Args:
        task_scores: Task name -> score.
        task_clusters: Task name -> cluster name.
        oracle_scores: Optional oracle scores for normalization.

    Returns:
        Dict of cluster_name -> {"mean_score", "normalized_score", "n_tasks"}.
    """
    cluster_scores: dict[str, list[float]] = defaultdict(list)
    cluster_normalized: dict[str, list[float]] = defaultdict(list)

    for task, score in task_scores.items():
        cluster = task_clusters.get(task, "unknown")
        cluster_scores[cluster].append(score)

        if oracle_scores and task in oracle_scores and oracle_scores[task] > 0:
            cluster_normalized[cluster].append(score / oracle_scores[task] * 100)

    result: dict[str, dict[str, float]] = {}
    for cluster, scores in cluster_scores.items():
        entry: dict[str, float] = {
            "mean_score": float(np.mean(scores)),
            "n_tasks": float(len(scores)),
        }
        if cluster in cluster_normalized:
            entry["normalized_score"] = float(np.mean(cluster_normalized[cluster]))
        result[cluster] = entry

    return result
