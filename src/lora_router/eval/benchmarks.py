"""FLAN v2 48-task benchmark for LoRA routing evaluation.

Implements the standard evaluation protocol used by LORAUTER, LoraRetriever, ARROW.
Supports three regimes: Non-OOD, Semi-OOD, OOD.

Two evaluation modes:
- Routing-only: measures routing accuracy, MRR, NDCG (no GPU needed)
- Full: generates outputs with selected adapters, computes EM/ROUGE/BLEU (needs GPU)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lora_router.eval.metrics import (
    mean_reciprocal_rank,
    ndcg,
    normalized_oracle_score,
    routing_accuracy,
    routing_accuracy_at_k,
)
from lora_router.registry import AdapterRegistry
from lora_router.strategies.base import BaseStrategy
from lora_router.types import AdapterInfo

# -- Data models --


@dataclass
class TaskConfig:
    """Configuration for a single benchmark task."""

    name: str
    hf_id: str
    description: str = ""
    cluster: str = ""
    metric: str = "exact_match"


@dataclass
class BenchmarkSample:
    """A single test sample for evaluation."""

    task: str
    input_text: str
    target_text: str
    metric: str = "exact_match"
    cluster: str = ""


@dataclass
class TaskResult:
    """Evaluation result for a single task."""

    task: str
    cluster: str
    metric_type: str
    # Routing metrics
    routing_accuracy_top1: float = 0.0
    routing_accuracy_top3: float = 0.0
    mrr: float = 0.0
    # Task metric (EM/ROUGE/BLEU) - only set in full eval mode
    task_score: float | None = None
    oracle_score: float | None = None
    n_samples: int = 0


@dataclass
class BenchmarkResult:
    """Complete benchmark result across all tasks and regimes."""

    regime: str  # non_ood, semi_ood, ood
    strategy_name: str
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    # Aggregate routing metrics
    routing_accuracy_top1: float = 0.0
    routing_accuracy_top3: float = 0.0
    mrr: float = 0.0
    ndcg_at5: float = 0.0
    # Normalized oracle score (only in full eval mode)
    normalized_oracle: float | None = None
    # Per-cluster breakdown
    cluster_results: dict[str, dict[str, float]] = field(default_factory=dict)
    # Timing
    total_routing_time_ms: float = 0.0
    avg_routing_time_ms: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON export."""
        return {
            "regime": self.regime,
            "strategy": self.strategy_name,
            "routing_accuracy_top1": round(self.routing_accuracy_top1, 4),
            "routing_accuracy_top3": round(self.routing_accuracy_top3, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at5": round(self.ndcg_at5, 4),
            "normalized_oracle": round(self.normalized_oracle, 2) if self.normalized_oracle else None,
            "avg_routing_time_ms": round(self.avg_routing_time_ms, 2),
            "n_samples": self.n_samples,
            "cluster_results": self.cluster_results,
            "task_results": {
                name: {
                    "cluster": tr.cluster,
                    "accuracy_top1": round(tr.routing_accuracy_top1, 4),
                    "accuracy_top3": round(tr.routing_accuracy_top3, 4),
                    "mrr": round(tr.mrr, 4),
                    "task_score": round(tr.task_score, 4) if tr.task_score is not None else None,
                    "oracle_score": round(tr.oracle_score, 4) if tr.oracle_score is not None else None,
                    "n_samples": tr.n_samples,
                }
                for name, tr in self.task_results.items()
            },
        }


# -- Benchmark config loader --


@dataclass
class FlanV2Config:
    """Parsed FLAN v2 benchmark configuration."""

    name: str
    base_model: str
    adapter_prefix: str
    tasks: list[TaskConfig]
    # Cluster -> metric mapping
    cluster_metrics: dict[str, str]
    # Task -> cluster mapping
    task_cluster_map: dict[str, str]
    # Baselines for comparison
    baselines: dict[str, dict[str, float]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> FlanV2Config:
        """Load benchmark configuration from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        tasks: list[TaskConfig] = []
        cluster_metrics: dict[str, str] = {}
        task_cluster_map: dict[str, str] = {}

        for cluster_name, cluster_data in raw.get("clusters", {}).items():
            metric = cluster_data.get("metric", "exact_match")
            cluster_metrics[cluster_name] = metric

            for task_entry in cluster_data.get("tasks", []):
                task = TaskConfig(
                    name=task_entry["name"],
                    hf_id=task_entry["hf_id"],
                    description=task_entry.get("description", ""),
                    cluster=cluster_name,
                    metric=metric,
                )
                tasks.append(task)
                task_cluster_map[task_entry["name"]] = cluster_name

        baselines = raw.get("baselines", {})

        return cls(
            name=raw.get("name", "flan_v2"),
            base_model=raw.get("base_model", ""),
            adapter_prefix=raw.get("adapter_prefix", ""),
            tasks=tasks,
            cluster_metrics=cluster_metrics,
            task_cluster_map=task_cluster_map,
            baselines=baselines,
        )


# -- Main benchmark class --


class FlanV2Benchmark:
    """FLAN v2 48-task routing benchmark.

    Evaluates routing quality across three regimes:
    - Non-OOD: All adapters available, ground-truth in pool
    - Semi-OOD: Adapter removed, task data remains for representation
    - OOD: Ground-truth adapter fully removed

    Usage:
        config = FlanV2Config.from_yaml("benchmarks/configs/flan_v2.yaml")
        benchmark = FlanV2Benchmark(config)
        benchmark.load_test_data("benchmarks/data/combined_test.json")
        result = benchmark.evaluate_routing(strategy, registry, regime="ood")
    """

    def __init__(self, config: FlanV2Config) -> None:
        self.config = config
        self.samples: list[BenchmarkSample] = []
        self._task_samples: dict[str, list[BenchmarkSample]] = defaultdict(list)

    @property
    def n_tasks(self) -> int:
        return len(self.config.tasks)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def task_names(self) -> list[str]:
        return [t.name for t in self.config.tasks]

    # -- Data loading --

    def load_test_data(self, path: str | Path) -> int:
        """Load test data from a JSON file (LoraRetriever format).

        Expected format: list of dicts with keys:
        - inputs: str (input text)
        - targets: str (target text)
        - task: str (task name matching adapter name)
        - domain: str (cluster/domain name)
        - metric: str (metric type)

        Returns:
            Number of samples loaded.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        self.samples.clear()
        self._task_samples.clear()

        valid_tasks = set(self.task_names)

        for entry in data:
            task = entry.get("task", "")
            if task not in valid_tasks:
                continue

            sample = BenchmarkSample(
                task=task,
                input_text=entry.get("inputs", ""),
                target_text=entry.get("targets", ""),
                metric=entry.get("metric", "exact_match"),
                cluster=self.config.task_cluster_map.get(task, ""),
            )
            self.samples.append(sample)
            self._task_samples[task].append(sample)

        return len(self.samples)

    def load_synthetic_data(self, samples_per_task: int = 50) -> int:
        """Generate synthetic test data for routing-only evaluation.

        Creates test samples using adapter descriptions as queries.
        Useful for testing routing logic without real FLAN v2 data.

        Returns:
            Number of samples generated.
        """
        self.samples.clear()
        self._task_samples.clear()

        for task_config in self.config.tasks:
            for i in range(samples_per_task):
                sample = BenchmarkSample(
                    task=task_config.name,
                    input_text=f"[{task_config.name}] {task_config.description} - sample {i}",
                    target_text=f"target_{task_config.name}_{i}",
                    metric=task_config.metric,
                    cluster=task_config.cluster,
                )
                self.samples.append(sample)
                self._task_samples[task_config.name].append(sample)

        return len(self.samples)

    # -- Registry helpers --

    def build_registry(
        self,
        adapter_dir: str | Path | None = None,
        embedder: Any = None,
        use_descriptions: bool = True,
    ) -> AdapterRegistry:
        """Build an AdapterRegistry from the benchmark tasks.

        If adapter_dir is provided, sets adapter paths to local files.
        If use_descriptions is True, uses task descriptions for embeddings.

        Args:
            adapter_dir: Local directory containing downloaded adapters.
            embedder: Embedding model for the registry.
            use_descriptions: Whether to use descriptions for embedding text.

        Returns:
            Populated AdapterRegistry.
        """
        registry = AdapterRegistry(embedder=embedder)
        adapter_dir = Path(adapter_dir) if adapter_dir else None

        for task_config in self.config.tasks:
            path = str(adapter_dir / task_config.name) if adapter_dir else task_config.hf_id
            adapter = AdapterInfo(
                name=task_config.name,
                path=path,
                description=task_config.description if use_descriptions else "",
                domain=task_config.cluster,
            )
            registry.register(adapter)

        return registry

    def get_ood_registry(
        self,
        full_registry: AdapterRegistry,
        exclude_task: str,
    ) -> AdapterRegistry:
        """Create a registry with the ground-truth adapter removed (OOD setting).

        Args:
            full_registry: Complete registry with all adapters.
            exclude_task: Task name to exclude.

        Returns:
            New registry without the excluded adapter.
        """
        return full_registry.exclude({exclude_task})

    # -- Routing evaluation (no GPU needed) --

    def evaluate_routing(
        self,
        strategy: BaseStrategy,
        registry: AdapterRegistry,
        regime: str = "non_ood",
        top_k: int = 5,
        batch_size: int = 32,
    ) -> BenchmarkResult:
        """Evaluate routing quality without running inference.

        Measures: routing accuracy (top-1, top-3), MRR, NDCG.
        This is the fast evaluation mode - no GPU required.

        Args:
            strategy: Routing strategy to evaluate.
            registry: Adapter registry.
            regime: Evaluation regime ("non_ood", "semi_ood", "ood").
            top_k: Number of adapters to retrieve per query.
            batch_size: Batch size for routing.

        Returns:
            BenchmarkResult with routing metrics.
        """
        if not self.samples:
            raise ValueError("No test data loaded. Call load_test_data() or load_synthetic_data() first.")

        # Collect per-task predictions
        task_predictions_top1: dict[str, list[str]] = defaultdict(list)
        task_predictions_ranked: dict[str, list[list[str]]] = defaultdict(list)
        task_ground_truth: dict[str, list[str]] = defaultdict(list)

        total_routing_time = 0.0

        # Process by task for OOD regime
        for task_config in self.config.tasks:
            task_name = task_config.name
            task_samples = self._task_samples.get(task_name, [])
            if not task_samples:
                continue

            # In OOD mode, remove the ground-truth adapter
            if regime == "ood":
                eval_registry = self.get_ood_registry(registry, task_name)
            else:
                eval_registry = registry

            queries = [s.input_text for s in task_samples]

            # Route in batches
            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size]
                start = time.perf_counter()
                batch_results = strategy.route_batch(batch, eval_registry, top_k=top_k)
                total_routing_time += (time.perf_counter() - start) * 1000

                for selections in batch_results:
                    if selections:
                        task_predictions_top1[task_name].append(selections[0].adapter_name)
                        task_predictions_ranked[task_name].append(
                            [s.adapter_name for s in selections]
                        )
                    else:
                        task_predictions_top1[task_name].append("")
                        task_predictions_ranked[task_name].append([])

            task_ground_truth[task_name] = [task_name] * len(task_samples)

        # Compute per-task metrics
        task_results: dict[str, TaskResult] = {}
        all_preds_top1: list[str] = []
        all_preds_ranked: list[list[str]] = []
        all_gt: list[str] = []

        for task_config in self.config.tasks:
            task_name = task_config.name
            preds_top1 = task_predictions_top1.get(task_name, [])
            preds_ranked = task_predictions_ranked.get(task_name, [])
            gt = task_ground_truth.get(task_name, [])

            if not preds_top1:
                continue

            tr = TaskResult(
                task=task_name,
                cluster=task_config.cluster,
                metric_type=task_config.metric,
                routing_accuracy_top1=routing_accuracy(preds_top1, gt),
                routing_accuracy_top3=routing_accuracy_at_k(preds_ranked, gt, k=3),
                mrr=mean_reciprocal_rank(preds_ranked, gt),
                n_samples=len(preds_top1),
            )
            task_results[task_name] = tr

            all_preds_top1.extend(preds_top1)
            all_preds_ranked.extend(preds_ranked)
            all_gt.extend(gt)

        # Compute aggregate metrics
        result = BenchmarkResult(
            regime=regime,
            strategy_name=strategy.name,
            task_results=task_results,
            routing_accuracy_top1=routing_accuracy(all_preds_top1, all_gt),
            routing_accuracy_top3=routing_accuracy_at_k(all_preds_ranked, all_gt, k=3),
            mrr=mean_reciprocal_rank(all_preds_ranked, all_gt),
            ndcg_at5=ndcg(all_preds_ranked, all_gt, k=5),
            total_routing_time_ms=total_routing_time,
            avg_routing_time_ms=total_routing_time / len(self.samples) if self.samples else 0.0,
            n_samples=len(all_preds_top1),
        )

        # Compute per-cluster routing accuracy
        cluster_preds: dict[str, list[str]] = defaultdict(list)
        cluster_gt: dict[str, list[str]] = defaultdict(list)
        for task_name, tr in task_results.items():
            cluster = tr.cluster
            cluster_preds[cluster].extend(task_predictions_top1.get(task_name, []))
            cluster_gt[cluster].extend(task_ground_truth.get(task_name, []))

        for cluster in cluster_preds:
            result.cluster_results[cluster] = {
                "routing_accuracy_top1": routing_accuracy(
                    cluster_preds[cluster], cluster_gt[cluster]
                ),
                "n_samples": float(len(cluster_preds[cluster])),
            }

        return result

    def evaluate_full(
        self,
        strategy: BaseStrategy,
        registry: AdapterRegistry,
        inference_fn: Any,
        oracle_scores: dict[str, float] | None = None,
        regime: str = "non_ood",
        top_k: int = 3,
    ) -> BenchmarkResult:
        """Full evaluation with inference and task metrics.

        Requires GPU and a loaded model. The inference_fn is called with:
            inference_fn(input_text, adapter_name) -> str

        This computes EM/ROUGE/BLEU per task and the normalized oracle score.

        Args:
            strategy: Routing strategy.
            registry: Adapter registry.
            inference_fn: Callable(input_text, adapter_name) -> generated_text.
            oracle_scores: Pre-computed oracle scores per task. If None, computes them.
            regime: Evaluation regime.
            top_k: Number of adapters for composition.

        Returns:
            BenchmarkResult with full metrics including normalized oracle score.
        """
        # First get routing results
        result = self.evaluate_routing(
            strategy, registry, regime=regime, top_k=top_k
        )

        # Then run inference for each task
        task_scores: dict[str, float] = {}

        for task_config in self.config.tasks:
            task_name = task_config.name
            task_samples = self._task_samples.get(task_name, [])
            if not task_samples:
                continue

            # Determine which adapter to use from routing
            tr = result.task_results.get(task_name)
            if not tr:
                continue

            # Get the top-1 predicted adapter for this task
            if regime == "ood":
                eval_registry = self.get_ood_registry(registry, task_name)
            else:
                eval_registry = registry

            # Route a representative sample to get the predicted adapter
            selections = strategy.route(
                task_samples[0].input_text, eval_registry, top_k=1
            )
            predicted_adapter = selections[0].adapter_name if selections else task_name

            # Run inference
            predictions = []
            references = []
            for sample in task_samples:
                output = inference_fn(sample.input_text, predicted_adapter)
                predictions.append(output)
                references.append(sample.target_text)

            # Compute task metric
            score = _compute_task_metric(
                predictions, references, task_config.metric
            )
            task_scores[task_name] = score
            tr.task_score = score

        # Compute oracle scores if not provided
        if oracle_scores is None:
            oracle_scores = {}
            for task_config in self.config.tasks:
                task_name = task_config.name
                task_samples = self._task_samples.get(task_name, [])
                if not task_samples:
                    continue
                oracle_preds = []
                oracle_refs = []
                for sample in task_samples:
                    output = inference_fn(sample.input_text, task_name)
                    oracle_preds.append(output)
                    oracle_refs.append(sample.target_text)
                oracle_scores[task_name] = _compute_task_metric(
                    oracle_preds, oracle_refs, task_config.metric
                )

        # Set oracle scores on task results
        for task_name, oracle in oracle_scores.items():
            if task_name in result.task_results:
                result.task_results[task_name].oracle_score = oracle

        # Compute normalized oracle score
        result.normalized_oracle = normalized_oracle_score(task_scores, oracle_scores)

        # Update cluster results with task scores
        cluster_task_scores: dict[str, list[float]] = defaultdict(list)
        cluster_oracle_scores: dict[str, list[float]] = defaultdict(list)
        for task_name, score in task_scores.items():
            cluster = self.config.task_cluster_map.get(task_name, "")
            cluster_task_scores[cluster].append(score)
            if task_name in oracle_scores:
                cluster_oracle_scores[cluster].append(oracle_scores[task_name])

        for cluster in cluster_task_scores:
            if cluster in result.cluster_results:
                scores = cluster_task_scores[cluster]
                result.cluster_results[cluster]["mean_task_score"] = float(np.mean(scores))
                if cluster in cluster_oracle_scores:
                    oracles = cluster_oracle_scores[cluster]
                    norm = [s / o * 100 for s, o in zip(scores, oracles) if o > 0]
                    if norm:
                        result.cluster_results[cluster]["normalized_score"] = float(np.mean(norm))

        return result

    # -- Result I/O --

    def save_results(self, result: BenchmarkResult, path: str | Path) -> None:
        """Save benchmark results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    @staticmethod
    def load_results(path: str | Path) -> dict[str, Any]:
        """Load benchmark results from JSON."""
        with open(path) as f:
            return json.load(f)

    def compare_to_baselines(self, result: BenchmarkResult) -> dict[str, dict[str, float]]:
        """Compare result against published baselines.

        Returns:
            Dict of baseline_name -> {delta_top1, delta_ood, etc.}
        """
        comparisons: dict[str, dict[str, float]] = {}
        regime = result.regime

        for name, baseline in self.config.baselines.items():
            comp: dict[str, float] = {}
            baseline_score = baseline.get(regime)
            if baseline_score is not None and result.normalized_oracle is not None:
                comp["baseline_normalized_oracle"] = baseline_score
                comp["our_normalized_oracle"] = result.normalized_oracle
                comp["delta"] = result.normalized_oracle - baseline_score
            comparisons[name] = comp

        return comparisons


# -- Task metric helpers --


def _compute_task_metric(
    predictions: list[str],
    references: list[str],
    metric_type: str,
) -> float:
    """Compute task-level metric (EM, ROUGE, or BLEU).

    Args:
        predictions: Generated outputs.
        references: Ground-truth targets.
        metric_type: One of "exact_match", "rouge", "bleu".

    Returns:
        Score as a float (0-100 scale).
    """
    if metric_type == "exact_match":
        return _exact_match(predictions, references)
    elif metric_type == "rouge":
        return _rouge_score(predictions, references)
    elif metric_type == "bleu":
        return _bleu_score(predictions, references)
    else:
        return _exact_match(predictions, references)


def _exact_match(predictions: list[str], references: list[str]) -> float:
    """Case-insensitive exact match after stripping."""
    if not predictions:
        return 0.0
    correct = 0
    for pred, ref in zip(predictions, references):
        # Strip and normalize like LoraRetriever
        pred_clean = pred.strip().lower().rstrip(".")
        ref_clean = ref.split("\n\n")[0].strip().lower().rstrip(".")
        if pred_clean == ref_clean:
            correct += 1
    return round(correct / len(predictions) * 100, 1)


def _rouge_score(predictions: list[str], references: list[str]) -> float:
    """ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            scores.append(result["rougeL"].fmeasure)
        return round(float(np.mean(scores)) * 100, 1) if scores else 0.0
    except ImportError:
        # Fallback: simple overlap ratio
        return _simple_rouge_fallback(predictions, references)


def _simple_rouge_fallback(predictions: list[str], references: list[str]) -> float:
    """Simple token overlap as ROUGE fallback when rouge_score not installed."""
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if not ref_tokens:
            scores.append(0.0)
            continue
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return round(float(np.mean(scores)) * 100, 1) if scores else 0.0


def _bleu_score(predictions: list[str], references: list[str]) -> float:
    """Sentence-level BLEU score."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smooth = SmoothingFunction().method1
        scores = []
        for pred, ref in zip(predictions, references):
            ref_tokens = ref.split()
            pred_tokens = pred.split()
            if not ref_tokens or not pred_tokens:
                scores.append(0.0)
                continue
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
            scores.append(score)
        return round(float(np.mean(scores)) * 100, 1) if scores else 0.0
    except ImportError:
        return 0.0
