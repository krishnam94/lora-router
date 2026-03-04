"""End-to-end GPU benchmark runner for lora-router.

Runs routing-only and full (with inference) evaluation across strategies
and regimes, then generates a comparison report against published baselines.

Usage:
    # Dry run - verify config, no execution
    python scripts/run_full_benchmark.py --dry-run

    # Routing-only eval (fast, no GPU model loading)
    python scripts/run_full_benchmark.py --routing-only \
        --adapter-dir /workspace/adapters/flan_v2

    # Full eval with inference for similarity strategy
    python scripts/run_full_benchmark.py --strategy similarity \
        --base-model /workspace/models/llama2-7b \
        --adapter-dir /workspace/adapters/flan_v2

    # Run everything - all strategies, both regimes
    python scripts/run_full_benchmark.py --all \
        --base-model /workspace/models/llama2-7b \
        --adapter-dir /workspace/adapters/flan_v2

    # Resume from cached oracle scores
    python scripts/run_full_benchmark.py --all \
        --base-model /workspace/models/llama2-7b \
        --adapter-dir /workspace/adapters/flan_v2 \
        --oracle-cache benchmarks/results/oracle_scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_header(text: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def print_result_summary(result: dict[str, object], baselines: dict[str, dict[str, float]]) -> None:
    """Print a formatted result summary with baseline comparison."""
    regime = result["regime"]
    strategy = result["strategy"]

    print(f"\n  Strategy: {strategy} | Regime: {regime}")
    print(f"  {'─' * 50}")
    print(f"  Routing Accuracy (top-1): {result['routing_accuracy_top1']:.1%}")
    print(f"  Routing Accuracy (top-3): {result['routing_accuracy_top3']:.1%}")
    print(f"  MRR:                      {result['mrr']:.4f}")
    print(f"  NDCG@5:                   {result['ndcg_at5']:.4f}")
    print(f"  Avg Routing Latency:      {result['avg_routing_time_ms']:.2f}ms")

    norm_oracle = result.get("normalized_oracle")
    if norm_oracle is not None:
        print(f"  Normalized Oracle Score:  {norm_oracle:.1f}")

        # Compare against baselines
        print(f"\n  {'─' * 50}")
        print(f"  {'Baseline':20s} {'Score':>8s} {'Delta':>8s} {'Status':>10s}")
        print(f"  {'─' * 50}")
        for name, scores in sorted(baselines.items()):
            baseline_score = scores.get(regime)
            if baseline_score is not None:
                delta = norm_oracle - baseline_score
                status = "BEAT" if delta > 0 else "BEHIND"
                marker = "+" if delta > 0 else ""
                print(f"  {name:20s} {baseline_score:8.1f} {marker}{delta:7.1f} {status:>10s}")

    # Per-cluster breakdown
    clusters = result.get("cluster_results", {})
    if clusters:
        print("\n  Per-cluster routing accuracy:")
        for cluster, metrics in sorted(clusters.items()):
            acc = metrics.get("routing_accuracy_top1", 0)
            n = int(metrics.get("n_samples", 0))
            norm = metrics.get("normalized_score")
            norm_str = f"  (norm: {norm:.1f})" if norm is not None else ""
            print(f"    {cluster:30s} {acc:.1%} ({n} samples){norm_str}")


def compute_oracle_scores(
    benchmark: object,
    inference_fn: object,
) -> dict[str, float]:
    """Compute oracle scores (best possible per-task score with ground-truth adapter).

    This is expensive - one inference pass per task. Cache the results.
    """
    from lora_router.eval.benchmarks import _compute_task_metric

    oracle_scores: dict[str, float] = {}

    for task_config in benchmark.config.tasks:  # type: ignore[attr-defined]
        task_name = task_config.name
        task_samples = benchmark._task_samples.get(task_name, [])  # type: ignore[attr-defined]
        if not task_samples:
            continue

        print(f"    Computing oracle for {task_name} ({len(task_samples)} samples)...", end=" ", flush=True)
        start = time.perf_counter()

        predictions = []
        references = []
        for sample in task_samples:
            output = inference_fn(sample.input_text, task_name)  # type: ignore[operator]
            predictions.append(output)
            references.append(sample.target_text)

        score = _compute_task_metric(predictions, references, task_config.metric)
        oracle_scores[task_name] = score
        elapsed = time.perf_counter() - start
        print(f"{score:.1f} ({elapsed:.1f}s)")

    return oracle_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lora-router GPU benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Verify config without running")
    mode.add_argument("--routing-only", action="store_true", help="Routing eval only (no GPU model)")
    mode.add_argument("--all", action="store_true", help="Full eval with all strategies")

    # Model / data
    parser.add_argument(
        "--base-model", type=str, default="meta-llama/Llama-2-7b-hf",
        help="Base model path or HF ID",
    )
    parser.add_argument(
        "--adapter-dir", type=str, default="adapters/flan_v2",
        help="Directory with downloaded LoRA adapters",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to test data JSON. Auto-detects benchmarks/data/combined_test.json",
    )
    parser.add_argument(
        "--config", type=str, default="benchmarks/configs/flan_v2.yaml",
        help="Benchmark config YAML",
    )

    # Strategy
    parser.add_argument(
        "--strategy", type=str, default="similarity",
        help="Strategy: similarity, ensemble, classifier (comma-separated for multiple)",
    )
    parser.add_argument(
        "--regime", type=str, default="non_ood,ood",
        help="Regime(s): non_ood, ood (comma-separated)",
    )

    # Performance
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens for generation")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token")

    # Caching
    parser.add_argument(
        "--oracle-cache", type=str, default=None,
        help="Path to cached oracle scores JSON (saves ~30 min on reruns)",
    )

    # Output
    parser.add_argument(
        "--results-dir", type=str, default="benchmarks/results",
        help="Directory to save results",
    )
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--plots", action="store_true", help="Generate plots")

    args = parser.parse_args()

    # Resolve project root
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect test data
    data_path = None
    if args.data:
        data_path = Path(args.data)
    else:
        default_data = project_root / "benchmarks" / "data" / "combined_test.json"
        if default_data.exists():
            data_path = default_data

    # Load config
    print_header("lora-router GPU Benchmark")
    print(f"\n  Config:     {config_path}")
    print(f"  Base model: {args.base_model}")
    print(f"  Adapters:   {args.adapter_dir}")
    print(f"  Test data:  {data_path or 'synthetic'}")
    print(f"  Results:    {results_dir}")

    from lora_router.eval.benchmarks import FlanV2Benchmark, FlanV2Config

    config = FlanV2Config.from_yaml(str(config_path))
    print(f"\n  Tasks: {len(config.tasks)} | Clusters: {len(config.cluster_metrics)}")

    benchmark = FlanV2Benchmark(config)

    # Load test data
    if data_path and data_path.exists():
        n = benchmark.load_test_data(str(data_path))
        print(f"  Loaded {n} test samples")
    else:
        n = benchmark.load_synthetic_data(samples_per_task=50)
        print(f"  Generated {n} synthetic samples")

    # Dry run - just verify config and exit
    if args.dry_run:
        print_header("Dry Run - Config Verified")
        print(f"\n  Tasks:   {benchmark.n_tasks}")
        print(f"  Samples: {benchmark.n_samples}")
        print(f"  Regimes: {args.regime}")
        print(f"  Strategy: {args.strategy}")
        print("\n  Baselines:")
        for name, scores in config.baselines.items():
            parts = [f"{k}={v}" for k, v in scores.items() if k != "source"]
            print(f"    {name:20s} {', '.join(parts)}")
        print("\n  Everything looks good. Remove --dry-run to execute.")
        return

    # Build registry
    print_header("Building Registry")

    adapter_dir = Path(args.adapter_dir)
    use_local = adapter_dir.exists() and any(adapter_dir.iterdir()) if adapter_dir.exists() else False

    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("  WARNING: sentence-transformers not available")
        embedder = None

    registry = benchmark.build_registry(
        adapter_dir=str(adapter_dir) if use_local else None,
        embedder=embedder,
    )
    print(f"  Registered {registry.size} adapters")
    print(f"  Local adapters: {'yes' if use_local else 'no (using HF IDs)'}")

    # Parse strategies and regimes
    strategy_names = [s.strip() for s in args.strategy.split(",")]
    regimes = [r.strip() for r in args.regime.split(",")]

    if args.all:
        strategy_names = ["similarity", "ensemble"]
        regimes = ["non_ood", "ood"]

    # Build strategies
    from lora_router.strategies.similarity import SimilarityStrategy

    def build_strategy(name: str) -> object:
        if name == "similarity":
            return SimilarityStrategy(encoder_name="all-MiniLM-L6-v2")
        elif name == "classifier":
            from lora_router.strategies.classifier import ClassifierStrategy
            s = ClassifierStrategy()
            s.train(registry)
            return s
        elif name == "ensemble":
            from lora_router.strategies.ensemble import EnsembleStrategy
            sim = SimilarityStrategy(encoder_name="all-MiniLM-L6-v2")
            strategies = [(sim, 1.0)]
            try:
                from lora_router.strategies.seqr import SEQRStrategy
                seqr = SEQRStrategy()
                strategies.append((seqr, 0.5))
            except Exception:
                pass
            return EnsembleStrategy(strategies=strategies)
        else:
            raise ValueError(f"Unknown strategy: {name}")

    # Phase 1: Routing-only evaluation
    print_header("Phase 1: Routing-Only Evaluation")
    all_results = []

    for strategy_name in strategy_names:
        strategy = build_strategy(strategy_name)
        for regime in regimes:
            print(f"\n  Running {strategy_name} / {regime}...")
            start = time.perf_counter()
            result = benchmark.evaluate_routing(
                strategy=strategy,  # type: ignore[arg-type]
                registry=registry,
                regime=regime,
                top_k=5,
            )
            elapsed = time.perf_counter() - start
            print(f"  Done in {elapsed:.1f}s")
            print_result_summary(result.to_dict(), config.baselines)

            # Save
            result_path = results_dir / f"{strategy_name}_{regime}_routing.json"
            benchmark.save_results(result, str(result_path))
            all_results.append(result)

    if args.routing_only:
        print_header("Routing-Only Benchmark Complete")
        print(f"\n  Results saved to {results_dir}/")
        _generate_outputs(args, all_results, config, results_dir)
        return

    # Phase 2: Full evaluation with inference
    print_header("Phase 2: Full Evaluation (GPU Inference)")

    # Load inference engine
    from lora_router.inference.engine import InferenceEngine

    engine = InferenceEngine(
        base_model=args.base_model,
        load_in_8bit=args.load_in_8bit,
        max_new_tokens=args.max_new_tokens,
        token=args.hf_token,
    )

    print("\n  Loading base model...")
    start = time.perf_counter()
    engine.load_base_model()
    print(f"  Base model loaded in {time.perf_counter() - start:.1f}s")

    mem = engine.gpu_memory_info()
    if "allocated_gb" in mem:
        print(f"  GPU memory: {mem['allocated_gb']:.1f}GB / {mem['total_gb']:.1f}GB")

    # Load all adapters
    task_names = [t.name for t in config.tasks]
    print(f"\n  Loading {len(task_names)} adapters...")
    start = time.perf_counter()
    loaded = engine.load_adapters_from_dir(str(adapter_dir), task_names)
    print(f"  Loaded {loaded}/{len(task_names)} adapters in {time.perf_counter() - start:.1f}s")

    mem = engine.gpu_memory_info()
    if "allocated_gb" in mem:
        print(f"  GPU memory: {mem['allocated_gb']:.1f}GB / {mem['total_gb']:.1f}GB")

    inference_fn = engine.create_inference_fn()

    # Load or compute oracle scores
    oracle_scores = None
    oracle_cache_path = results_dir / "oracle_scores.json"

    if args.oracle_cache:
        oracle_cache_path = Path(args.oracle_cache)

    if oracle_cache_path.exists():
        print(f"\n  Loading cached oracle scores from {oracle_cache_path}")
        with open(oracle_cache_path) as f:
            oracle_scores = json.load(f)
        print(f"  Loaded oracle scores for {len(oracle_scores)} tasks")
    else:
        print("\n  Computing oracle scores (this takes a while)...")
        start = time.perf_counter()
        oracle_scores = compute_oracle_scores(benchmark, inference_fn)
        elapsed = time.perf_counter() - start
        print(f"\n  Oracle scores computed in {elapsed:.1f}s")

        # Cache for future runs
        with open(oracle_cache_path, "w") as f:
            json.dump(oracle_scores, f, indent=2)
        print(f"  Cached to {oracle_cache_path}")

    # Run full evaluation
    full_results = []
    for strategy_name in strategy_names:
        strategy = build_strategy(strategy_name)
        for regime in regimes:
            print(f"\n  Running full eval: {strategy_name} / {regime}...")
            start = time.perf_counter()
            result = benchmark.evaluate_full(
                strategy=strategy,  # type: ignore[arg-type]
                registry=registry,
                inference_fn=inference_fn,
                oracle_scores=oracle_scores,
                regime=regime,
                top_k=3,
            )
            elapsed = time.perf_counter() - start
            print(f"  Done in {elapsed:.1f}s")
            print_result_summary(result.to_dict(), config.baselines)

            # Save
            result_path = results_dir / f"{strategy_name}_{regime}_full.json"
            benchmark.save_results(result, str(result_path))
            full_results.append(result)

    # Cleanup
    engine.unload()

    print_header("Benchmark Complete")
    print(f"\n  Results saved to {results_dir}/")

    # Final comparison table
    print("\n  Final Results vs Baselines:")
    print(f"  {'─' * 60}")
    print(f"  {'Method':25s} {'Non-OOD':>10s} {'OOD':>10s}")
    print(f"  {'─' * 60}")

    # Published baselines
    for name, scores in config.baselines.items():
        non_ood = scores.get("non_ood", "-")
        ood = scores.get("ood", "-")
        non_ood_str = f"{non_ood}" if isinstance(non_ood, (int, float)) else str(non_ood)
        ood_str = f"{ood}" if isinstance(ood, (int, float)) else str(ood)
        print(f"  {name:25s} {non_ood_str:>10s} {ood_str:>10s}")

    print(f"  {'─' * 60}")

    # Our results
    for result in full_results:
        d = result.to_dict()
        norm = d.get("normalized_oracle")
        regime = d["regime"]
        strategy = d["strategy"]
        label = f"ours ({strategy})"
        score_str = f"{norm:.1f}" if norm is not None else "-"
        if regime == "non_ood":
            print(f"  {label:25s} {score_str:>10s} {'':>10s}")
        else:
            print(f"  {label:25s} {'':>10s} {score_str:>10s}")

    _generate_outputs(args, full_results, config, results_dir)


def _generate_outputs(
    args: argparse.Namespace,
    results: list[object],
    config: object,
    results_dir: Path,
) -> None:
    """Generate report and plots if requested."""
    if args.report:
        from lora_router.eval.report import generate_markdown_report, save_report

        report_content = generate_markdown_report(
            results=[r.to_dict() for r in results],  # type: ignore[attr-defined]
            baselines=config.baselines,  # type: ignore[attr-defined]
        )
        report_path = results_dir / "benchmark_report.md"
        save_report(report_content, str(report_path))
        print(f"\n  Report: {report_path}")

    if args.plots:
        from lora_router.eval.report import generate_plots

        plot_paths = generate_plots(
            results=[r.to_dict() for r in results],  # type: ignore[attr-defined]
            output_dir=str(results_dir / "plots"),
            baselines=config.baselines,  # type: ignore[attr-defined]
        )
        for p in plot_paths:
            print(f"  Plot: {p}")


if __name__ == "__main__":
    main()
