"""Unit tests for FlanV2Benchmark and related classes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_router.eval.benchmarks import (
    BenchmarkResult,
    BenchmarkSample,
    FlanV2Benchmark,
    FlanV2Config,
    TaskConfig,
    TaskResult,
    _compute_task_metric,
    _exact_match,
    _simple_rouge_fallback,
)
from lora_router.strategies.similarity import SimilarityStrategy

# -- Fixtures --


@pytest.fixture
def flan_config_path(tmp_path: Path) -> Path:
    """Create a minimal FLAN v2 config YAML for testing."""
    config = {
        "name": "flan_v2_test",
        "base_model": "test-model",
        "adapter_prefix": "test",
        "clusters": {
            "sentiment": {
                "metric": "exact_match",
                "tasks": [
                    {"name": "sst2", "hf_id": "test/sst2", "description": "Sentiment classification"},
                    {"name": "imdb_reviews", "hf_id": "test/imdb", "description": "Movie review sentiment"},
                ],
            },
            "nli": {
                "metric": "exact_match",
                "tasks": [
                    {"name": "rte", "hf_id": "test/rte", "description": "Textual entailment"},
                    {"name": "cb", "hf_id": "test/cb", "description": "CommitmentBank NLI"},
                ],
            },
            "translation": {
                "metric": "bleu",
                "tasks": [
                    {"name": "wmt14_enfr", "hf_id": "test/wmt", "description": "English to French"},
                ],
            },
        },
        "baselines": {
            "lorauter": {"non_ood": 101.2, "ood": 88.4, "source": "test"},
            "lora_retriever": {"non_ood": 92.9, "ood": 83.2, "source": "test"},
        },
    }
    import yaml

    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def flan_config(flan_config_path: Path) -> FlanV2Config:
    return FlanV2Config.from_yaml(flan_config_path)


@pytest.fixture
def test_data_path(tmp_path: Path) -> Path:
    """Create test data JSON in LoraRetriever format."""
    samples = []
    tasks = {
        "sst2": ("sentiment", "exact_match"),
        "imdb_reviews": ("sentiment", "exact_match"),
        "rte": ("nli", "exact_match"),
        "cb": ("nli", "exact_match"),
        "wmt14_enfr": ("translation", "bleu"),
    }
    for task_name, (domain, metric) in tasks.items():
        for i in range(10):
            samples.append({
                "task": task_name,
                "domain": domain,
                "metric": metric,
                "inputs": f"Test input for {task_name} sample {i}",
                "targets": f"target_{task_name}_{i}",
            })

    path = tmp_path / "test_data.json"
    with open(path, "w") as f:
        json.dump(samples, f)
    return path


@pytest.fixture
def benchmark_with_data(flan_config: FlanV2Config, test_data_path: Path) -> FlanV2Benchmark:
    """Benchmark initialized with test config and data."""
    bm = FlanV2Benchmark(flan_config)
    bm.load_test_data(test_data_path)
    return bm


# -- FlanV2Config tests --


class TestFlanV2Config:
    def test_load_from_yaml(self, flan_config: FlanV2Config) -> None:
        assert flan_config.name == "flan_v2_test"
        assert len(flan_config.tasks) == 5
        assert flan_config.base_model == "test-model"

    def test_cluster_metrics(self, flan_config: FlanV2Config) -> None:
        assert flan_config.cluster_metrics["sentiment"] == "exact_match"
        assert flan_config.cluster_metrics["translation"] == "bleu"

    def test_task_cluster_map(self, flan_config: FlanV2Config) -> None:
        assert flan_config.task_cluster_map["sst2"] == "sentiment"
        assert flan_config.task_cluster_map["rte"] == "nli"
        assert flan_config.task_cluster_map["wmt14_enfr"] == "translation"

    def test_baselines(self, flan_config: FlanV2Config) -> None:
        assert flan_config.baselines["lorauter"]["ood"] == 88.4
        assert flan_config.baselines["lora_retriever"]["non_ood"] == 92.9

    def test_task_configs(self, flan_config: FlanV2Config) -> None:
        sst2 = [t for t in flan_config.tasks if t.name == "sst2"][0]
        assert sst2.hf_id == "test/sst2"
        assert sst2.cluster == "sentiment"
        assert sst2.metric == "exact_match"


# -- FlanV2Benchmark tests --


class TestFlanV2Benchmark:
    def test_init(self, flan_config: FlanV2Config) -> None:
        bm = FlanV2Benchmark(flan_config)
        assert bm.n_tasks == 5
        assert bm.n_samples == 0

    def test_load_test_data(self, flan_config: FlanV2Config, test_data_path: Path) -> None:
        bm = FlanV2Benchmark(flan_config)
        n = bm.load_test_data(test_data_path)
        assert n == 50  # 5 tasks * 10 samples
        assert bm.n_samples == 50

    def test_load_test_data_filters_unknown_tasks(
        self, flan_config: FlanV2Config, tmp_path: Path
    ) -> None:
        data = [
            {"task": "sst2", "domain": "sentiment", "metric": "em", "inputs": "x", "targets": "y"},
            {"task": "unknown_task", "domain": "?", "metric": "em", "inputs": "x", "targets": "y"},
        ]
        path = tmp_path / "mixed.json"
        with open(path, "w") as f:
            json.dump(data, f)

        bm = FlanV2Benchmark(flan_config)
        n = bm.load_test_data(path)
        assert n == 1

    def test_load_synthetic_data(self, flan_config: FlanV2Config) -> None:
        bm = FlanV2Benchmark(flan_config)
        n = bm.load_synthetic_data(samples_per_task=20)
        assert n == 100  # 5 tasks * 20
        assert bm.n_samples == 100

    def test_task_names(self, flan_config: FlanV2Config) -> None:
        bm = FlanV2Benchmark(flan_config)
        names = bm.task_names
        assert "sst2" in names
        assert "rte" in names
        assert len(names) == 5

    def test_build_registry(self, flan_config: FlanV2Config, mock_embedder) -> None:
        bm = FlanV2Benchmark(flan_config)
        registry = bm.build_registry(embedder=mock_embedder)
        assert registry.size == 5
        assert registry.has("sst2")
        assert registry.has("wmt14_enfr")

    def test_build_registry_with_adapter_dir(
        self, flan_config: FlanV2Config, tmp_path: Path
    ) -> None:
        bm = FlanV2Benchmark(flan_config)
        registry = bm.build_registry(adapter_dir=tmp_path)
        sst2 = registry.get("sst2")
        assert str(tmp_path / "sst2") == sst2.path

    def test_get_ood_registry(self, flan_config: FlanV2Config, mock_embedder) -> None:
        bm = FlanV2Benchmark(flan_config)
        full_registry = bm.build_registry(embedder=mock_embedder)
        ood_registry = bm.get_ood_registry(full_registry, "sst2")
        assert not ood_registry.has("sst2")
        assert ood_registry.has("imdb_reviews")
        assert ood_registry.size == 4


class TestBenchmarkEvaluation:
    def test_evaluate_routing_non_ood(self, benchmark_with_data, mock_embedder) -> None:
        bm = benchmark_with_data
        registry = bm.build_registry(embedder=mock_embedder)
        strategy = SimilarityStrategy(encoder=mock_embedder)

        result = bm.evaluate_routing(strategy, registry, regime="non_ood")

        assert result.regime == "non_ood"
        assert result.strategy_name == "SimilarityStrategy"
        assert result.n_samples == 50
        assert 0.0 <= result.routing_accuracy_top1 <= 1.0
        assert 0.0 <= result.routing_accuracy_top3 <= 1.0
        assert 0.0 <= result.mrr <= 1.0
        assert result.total_routing_time_ms > 0

    def test_evaluate_routing_ood(self, benchmark_with_data, mock_embedder) -> None:
        bm = benchmark_with_data
        registry = bm.build_registry(embedder=mock_embedder)
        strategy = SimilarityStrategy(encoder=mock_embedder)

        result = bm.evaluate_routing(strategy, registry, regime="ood")

        assert result.regime == "ood"
        assert result.n_samples == 50
        # In OOD, ground-truth adapter is removed, so top-1 accuracy should be 0
        assert result.routing_accuracy_top1 == 0.0

    def test_evaluate_routing_has_task_results(self, benchmark_with_data, mock_embedder) -> None:
        bm = benchmark_with_data
        registry = bm.build_registry(embedder=mock_embedder)
        strategy = SimilarityStrategy(encoder=mock_embedder)

        result = bm.evaluate_routing(strategy, registry, regime="non_ood")

        assert "sst2" in result.task_results
        assert "rte" in result.task_results
        tr = result.task_results["sst2"]
        assert tr.cluster == "sentiment"
        assert tr.n_samples == 10

    def test_evaluate_routing_has_cluster_results(self, benchmark_with_data, mock_embedder) -> None:
        bm = benchmark_with_data
        registry = bm.build_registry(embedder=mock_embedder)
        strategy = SimilarityStrategy(encoder=mock_embedder)

        result = bm.evaluate_routing(strategy, registry, regime="non_ood")

        assert "sentiment" in result.cluster_results
        assert "nli" in result.cluster_results
        assert result.cluster_results["sentiment"]["n_samples"] == 20.0  # sst2 + imdb

    def test_evaluate_routing_no_data_raises(self, flan_config, mock_embedder) -> None:
        bm = FlanV2Benchmark(flan_config)
        registry = bm.build_registry(embedder=mock_embedder)
        strategy = SimilarityStrategy(encoder=mock_embedder)

        with pytest.raises(ValueError, match="No test data"):
            bm.evaluate_routing(strategy, registry)


class TestBenchmarkResultSerialization:
    def test_to_dict(self) -> None:
        result = BenchmarkResult(
            regime="ood",
            strategy_name="test",
            routing_accuracy_top1=0.75,
            mrr=0.82,
            n_samples=100,
        )
        d = result.to_dict()
        assert d["regime"] == "ood"
        assert d["routing_accuracy_top1"] == 0.75
        assert d["mrr"] == 0.82
        assert d["n_samples"] == 100

    def test_save_and_load(self, benchmark_with_data, mock_embedder, tmp_path) -> None:
        bm = benchmark_with_data
        registry = bm.build_registry(embedder=mock_embedder)
        strategy = SimilarityStrategy(encoder=mock_embedder)

        result = bm.evaluate_routing(strategy, registry)
        path = tmp_path / "result.json"
        bm.save_results(result, path)

        loaded = FlanV2Benchmark.load_results(path)
        assert loaded["regime"] == "non_ood"
        assert loaded["n_samples"] == 50

    def test_compare_to_baselines(self, benchmark_with_data, mock_embedder) -> None:
        bm = benchmark_with_data
        registry = bm.build_registry(embedder=mock_embedder)
        strategy = SimilarityStrategy(encoder=mock_embedder)

        result = bm.evaluate_routing(strategy, registry, regime="ood")
        # Set a fake normalized oracle to test comparison
        result.normalized_oracle = 85.0

        comparisons = bm.compare_to_baselines(result)
        assert "lorauter" in comparisons
        assert comparisons["lorauter"]["baseline_normalized_oracle"] == 88.4
        assert comparisons["lorauter"]["delta"] == pytest.approx(-3.4)


# -- Task metric tests --


class TestTaskMetrics:
    def test_exact_match_perfect(self) -> None:
        preds = ["yes", "no", "yes"]
        refs = ["yes", "no", "yes"]
        assert _exact_match(preds, refs) == 100.0

    def test_exact_match_case_insensitive(self) -> None:
        preds = ["Yes", "NO"]
        refs = ["yes", "no"]
        assert _exact_match(preds, refs) == 100.0

    def test_exact_match_strips_period(self) -> None:
        preds = ["yes."]
        refs = ["yes"]
        assert _exact_match(preds, refs) == 100.0

    def test_exact_match_partial(self) -> None:
        preds = ["yes", "wrong"]
        refs = ["yes", "no"]
        assert _exact_match(preds, refs) == 50.0

    def test_exact_match_empty(self) -> None:
        assert _exact_match([], []) == 0.0

    def test_exact_match_ref_with_double_newline(self) -> None:
        """LoraRetriever truncates references at double newline."""
        preds = ["answer"]
        refs = ["answer\n\nextra context"]
        assert _exact_match(preds, refs) == 100.0

    def test_simple_rouge_fallback(self) -> None:
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = _simple_rouge_fallback(preds, refs)
        assert score == 100.0

    def test_simple_rouge_partial(self) -> None:
        preds = ["the cat sat"]
        refs = ["the cat sat on the mat"]
        score = _simple_rouge_fallback(preds, refs)
        assert 0.0 < score < 100.0

    def test_compute_task_metric_em(self) -> None:
        preds = ["yes", "no"]
        refs = ["yes", "no"]
        assert _compute_task_metric(preds, refs, "exact_match") == 100.0

    def test_compute_task_metric_unknown_defaults_em(self) -> None:
        preds = ["yes"]
        refs = ["yes"]
        assert _compute_task_metric(preds, refs, "unknown_metric") == 100.0


# -- TaskResult / TaskConfig tests --


class TestDataModels:
    def test_task_config(self) -> None:
        tc = TaskConfig(name="sst2", hf_id="test/sst2", cluster="sentiment")
        assert tc.name == "sst2"
        assert tc.metric == "exact_match"  # default

    def test_benchmark_sample(self) -> None:
        sample = BenchmarkSample(
            task="sst2",
            input_text="Great movie!",
            target_text="positive",
        )
        assert sample.metric == "exact_match"  # default

    def test_task_result(self) -> None:
        tr = TaskResult(
            task="sst2",
            cluster="sentiment",
            metric_type="exact_match",
            routing_accuracy_top1=0.8,
            n_samples=50,
        )
        assert tr.routing_accuracy_top1 == 0.8

    def test_benchmark_result_defaults(self) -> None:
        br = BenchmarkResult(regime="non_ood", strategy_name="test")
        assert br.n_samples == 0
        assert br.normalized_oracle is None
        assert br.cluster_results == {}
