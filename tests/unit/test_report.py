"""Unit tests for report generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from lora_router.eval.report import (
    generate_comparison_table,
    generate_markdown_report,
    save_report,
)


@pytest.fixture
def sample_results() -> list[dict]:
    """Two sample benchmark result dicts."""
    return [
        {
            "regime": "non_ood",
            "strategy": "SimilarityStrategy",
            "routing_accuracy_top1": 0.72,
            "routing_accuracy_top3": 0.90,
            "mrr": 0.81,
            "ndcg_at5": 0.85,
            "normalized_oracle": None,
            "avg_routing_time_ms": 2.5,
            "n_samples": 2400,
            "cluster_results": {
                "sentiment": {"routing_accuracy_top1": 0.85, "n_samples": 200},
                "nli": {"routing_accuracy_top1": 0.60, "n_samples": 500},
            },
            "task_results": {
                "sst2": {
                    "cluster": "sentiment",
                    "accuracy_top1": 0.90,
                    "accuracy_top3": 0.98,
                    "mrr": 0.93,
                    "task_score": None,
                    "oracle_score": None,
                    "n_samples": 50,
                },
            },
        },
        {
            "regime": "ood",
            "strategy": "SimilarityStrategy",
            "routing_accuracy_top1": 0.0,
            "routing_accuracy_top3": 0.65,
            "mrr": 0.42,
            "ndcg_at5": 0.50,
            "normalized_oracle": 85.0,
            "avg_routing_time_ms": 2.8,
            "n_samples": 2400,
            "cluster_results": {},
            "task_results": {},
        },
    ]


@pytest.fixture
def sample_baselines() -> dict:
    return {
        "lorauter": {"non_ood": 101.2, "ood": 88.4, "source": "arXiv:2601.21795"},
        "lora_retriever": {"non_ood": 92.9, "ood": 83.2, "source": "ACL 2024"},
    }


class TestMarkdownReport:
    def test_generates_markdown(self, sample_results) -> None:
        report = generate_markdown_report(sample_results)
        assert "# FLAN v2 Benchmark Results" in report
        assert "SimilarityStrategy" in report

    def test_includes_summary_table(self, sample_results) -> None:
        report = generate_markdown_report(sample_results)
        assert "Routing Accuracy (top-1)" in report
        assert "MRR" in report
        assert "72.0%" in report

    def test_includes_cluster_results(self, sample_results) -> None:
        report = generate_markdown_report(sample_results)
        assert "sentiment" in report
        assert "85.0%" in report

    def test_includes_task_results(self, sample_results) -> None:
        report = generate_markdown_report(sample_results)
        assert "sst2" in report

    def test_includes_baselines(self, sample_results, sample_baselines) -> None:
        report = generate_markdown_report(sample_results, baselines=sample_baselines)
        assert "Comparison to Baselines" in report
        assert "lorauter" in report
        assert "88.4%" in report

    def test_custom_title(self, sample_results) -> None:
        report = generate_markdown_report(sample_results, title="Custom Title")
        assert "# Custom Title" in report

    def test_empty_results(self) -> None:
        report = generate_markdown_report([])
        assert "# FLAN v2 Benchmark Results" in report


class TestComparisonTable:
    def test_generates_table(self, sample_results) -> None:
        table = generate_comparison_table(sample_results)
        assert "Strategy" in table
        assert "SimilarityStrategy" in table

    def test_empty_results(self) -> None:
        assert generate_comparison_table([]) == ""

    def test_custom_metric(self, sample_results) -> None:
        table = generate_comparison_table(sample_results, metric="mrr")
        assert "mrr" in table


class TestSaveReport:
    def test_save_to_file(self, tmp_path: Path) -> None:
        content = "# Test Report\n\nSome content."
        path = tmp_path / "reports" / "test.md"
        save_report(content, path)
        assert path.exists()
        assert path.read_text() == content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "report.md"
        save_report("test", path)
        assert path.exists()
