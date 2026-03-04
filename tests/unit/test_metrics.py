"""Unit tests for lora_router.eval.metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from lora_router.eval.metrics import (
    mean_reciprocal_rank,
    ndcg,
    normalized_oracle_score,
    per_cluster_scores,
    routing_accuracy,
    routing_accuracy_at_k,
)

pytestmark = pytest.mark.unit


class TestRoutingAccuracy:
    def test_perfect_accuracy(self):
        preds = ["code", "math", "creative"]
        truth = ["code", "math", "creative"]
        assert routing_accuracy(preds, truth) == 1.0

    def test_zero_accuracy(self):
        preds = ["code", "code", "code"]
        truth = ["math", "math", "math"]
        assert routing_accuracy(preds, truth) == 0.0

    def test_partial_accuracy(self):
        preds = ["code", "math", "creative"]
        truth = ["code", "creative", "creative"]
        assert routing_accuracy(preds, truth) == pytest.approx(2 / 3)

    def test_empty_predictions(self):
        assert routing_accuracy([], []) == 0.0


class TestRoutingAccuracyAtK:
    def test_perfect_at_k(self):
        preds = [["code", "math"], ["math", "code"], ["creative", "code"]]
        truth = ["code", "math", "creative"]
        assert routing_accuracy_at_k(preds, truth, k=2) == 1.0

    def test_imperfect_at_k(self):
        preds = [["code", "math"], ["creative", "code"]]
        truth = ["code", "math"]
        # First query: "code" in top-1 -> correct at k=1
        # Second query: "math" not in ["creative"] at k=1 -> incorrect
        assert routing_accuracy_at_k(preds, truth, k=1) == 0.5

    def test_k_larger_than_predictions(self):
        preds = [["code"]]
        truth = ["code"]
        assert routing_accuracy_at_k(preds, truth, k=10) == 1.0

    def test_empty_predictions(self):
        assert routing_accuracy_at_k([], [], k=3) == 0.0


class TestMeanReciprocalRank:
    def test_all_rank_one(self):
        """All correct at rank 1 gives MRR = 1.0."""
        preds = [["code", "math"], ["math", "code"]]
        truth = ["code", "math"]
        assert mean_reciprocal_rank(preds, truth) == 1.0

    def test_mixed_ranks(self):
        """MRR with mix of ranks 1 and 2."""
        preds = [["code", "math"], ["code", "math"]]
        truth = ["code", "math"]
        # Query 1: code at rank 1 -> 1/1, Query 2: math at rank 2 -> 1/2
        expected = (1.0 + 0.5) / 2
        assert mean_reciprocal_rank(preds, truth) == pytest.approx(expected)

    def test_not_in_predictions(self):
        """Ground truth not in predictions gives 0 for that query."""
        preds = [["code", "math"]]
        truth = ["creative"]
        assert mean_reciprocal_rank(preds, truth) == 0.0

    def test_empty(self):
        assert mean_reciprocal_rank([], []) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self):
        """Ground truth at rank 1 gives NDCG@k = 1.0."""
        preds = [["code", "math", "creative"]]
        truth = ["code"]
        assert ndcg(preds, truth, k=3) == pytest.approx(1.0)

    def test_ground_truth_at_rank_two(self):
        """Ground truth at rank 2 gives reduced NDCG."""
        preds = [["math", "code"]]
        truth = ["code"]
        # DCG = 1/log2(2+1) = 1/log2(3), IDCG = 1/log2(2) = 1.0
        expected = (1.0 / np.log2(3)) / 1.0
        assert ndcg(preds, truth, k=3) == pytest.approx(expected, abs=1e-6)

    def test_not_in_top_k(self):
        """Ground truth not in top-k gives NDCG = 0."""
        preds = [["math", "creative"]]
        truth = ["code"]
        assert ndcg(preds, truth, k=2) == 0.0

    def test_empty(self):
        assert ndcg([], [], k=5) == 0.0


class TestNormalizedOracleScore:
    def test_perfect_oracle(self):
        """Method matching oracle gives 100%."""
        method = {"task1": 0.9, "task2": 0.8}
        oracle = {"task1": 0.9, "task2": 0.8}
        assert normalized_oracle_score(method, oracle) == pytest.approx(100.0)

    def test_half_oracle(self):
        """Method at half oracle score gives 50%."""
        method = {"task1": 0.5}
        oracle = {"task1": 1.0}
        assert normalized_oracle_score(method, oracle) == pytest.approx(50.0)

    def test_exceeds_oracle(self):
        """Composition can exceed oracle (>100%)."""
        method = {"task1": 1.2}
        oracle = {"task1": 1.0}
        assert normalized_oracle_score(method, oracle) == pytest.approx(120.0)

    def test_zero_oracle_score_skipped(self):
        """Tasks with zero oracle score are skipped."""
        method = {"task1": 0.5, "task2": 0.3}
        oracle = {"task1": 1.0, "task2": 0.0}
        # Only task1 is counted: 0.5/1.0 * 100 = 50%
        assert normalized_oracle_score(method, oracle) == pytest.approx(50.0)

    def test_empty_inputs(self):
        assert normalized_oracle_score({}, {}) == 0.0
        assert normalized_oracle_score({"a": 0.5}, {}) == 0.0


class TestPerClusterScores:
    def test_basic_clustering(self):
        task_scores = {"t1": 0.9, "t2": 0.7, "t3": 0.8}
        task_clusters = {"t1": "code", "t2": "code", "t3": "math"}
        result = per_cluster_scores(task_scores, task_clusters)
        assert "code" in result
        assert "math" in result
        np.testing.assert_almost_equal(result["code"]["mean_score"], 0.8, decimal=5)
        np.testing.assert_almost_equal(result["math"]["mean_score"], 0.8, decimal=5)
        assert result["code"]["n_tasks"] == 2
        assert result["math"]["n_tasks"] == 1

    def test_with_oracle_scores(self):
        task_scores = {"t1": 0.9, "t2": 0.8}
        task_clusters = {"t1": "code", "t2": "code"}
        oracle_scores = {"t1": 1.0, "t2": 1.0}
        result = per_cluster_scores(task_scores, task_clusters, oracle_scores)
        assert "normalized_score" in result["code"]
        # (90 + 80) / 2 = 85
        np.testing.assert_almost_equal(
            result["code"]["normalized_score"], 85.0, decimal=3
        )

    def test_unknown_cluster(self):
        """Tasks without a cluster mapping go to 'unknown'."""
        task_scores = {"t1": 0.5}
        task_clusters = {}  # t1 not mapped
        result = per_cluster_scores(task_scores, task_clusters)
        assert "unknown" in result
        assert result["unknown"]["mean_score"] == 0.5

    def test_empty_inputs(self):
        result = per_cluster_scores({}, {})
        assert result == {}
