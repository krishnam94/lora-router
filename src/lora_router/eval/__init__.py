"""Evaluation metrics, benchmarks, and report generation."""

from lora_router.eval.benchmarks import (
    BenchmarkResult,
    BenchmarkSample,
    FlanV2Benchmark,
    FlanV2Config,
    TaskConfig,
    TaskResult,
)
from lora_router.eval.metrics import (
    mean_reciprocal_rank,
    ndcg,
    normalized_oracle_score,
    per_cluster_scores,
    routing_accuracy,
    routing_accuracy_at_k,
)
from lora_router.eval.report import (
    generate_comparison_table,
    generate_markdown_report,
    generate_plots,
    save_report,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSample",
    "FlanV2Benchmark",
    "FlanV2Config",
    "TaskConfig",
    "TaskResult",
    "generate_comparison_table",
    "generate_markdown_report",
    "generate_plots",
    "mean_reciprocal_rank",
    "ndcg",
    "normalized_oracle_score",
    "per_cluster_scores",
    "routing_accuracy",
    "routing_accuracy_at_k",
    "save_report",
]
