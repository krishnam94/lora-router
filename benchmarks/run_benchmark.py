"""Run FLAN v2 48-task benchmark.

Usage:
    # Routing-only evaluation (no GPU needed)
    python benchmarks/run_benchmark.py --config benchmarks/configs/flan_v2.yaml \
        --strategy similarity --regime non_ood --synthetic

    # With real test data
    python benchmarks/run_benchmark.py --config benchmarks/configs/flan_v2.yaml \
        --strategy similarity --data benchmarks/data/combined_test.json

    # Multiple strategies comparison
    python benchmarks/run_benchmark.py --config benchmarks/configs/flan_v2.yaml \
        --strategy similarity,seqr,ensemble --regime ood \
        --adapter-dir adapters/flan_v2

    # Generate report
    python benchmarks/run_benchmark.py --config benchmarks/configs/flan_v2.yaml \
        --strategy similarity --regime non_ood,ood --report benchmarks/results/report.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lora_router.eval.benchmarks import BenchmarkResult, FlanV2Benchmark, FlanV2Config
from lora_router.eval.report import generate_markdown_report, save_report
from lora_router.registry import AdapterRegistry
from lora_router.strategies.base import BaseStrategy
from lora_router.strategies.similarity import SimilarityStrategy


def build_strategy(
    name: str,
    registry: AdapterRegistry,
    encoder: str = "all-MiniLM-L6-v2",
) -> BaseStrategy:
    """Build a routing strategy by name."""
    if name == "similarity":
        return SimilarityStrategy(encoder_name=encoder)
    elif name == "seqr":
        from lora_router.strategies.seqr import SEQRStrategy

        return SEQRStrategy()
    elif name == "classifier":
        from lora_router.strategies.classifier import ClassifierStrategy

        strategy = ClassifierStrategy()
        strategy.train(registry)
        return strategy
    elif name == "ensemble":
        from lora_router.strategies.ensemble import EnsembleStrategy

        sim = SimilarityStrategy(encoder_name=encoder)
        strategies = [(sim, 1.0)]
        # Add SEQR if available
        try:
            from lora_router.strategies.seqr import SEQRStrategy

            seqr = SEQRStrategy()
            strategies.append((seqr, 0.5))
        except Exception:
            pass
        return EnsembleStrategy(strategies=strategies)
    else:
        raise ValueError(f"Unknown strategy: {name}. Use: similarity, seqr, classifier, ensemble")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FLAN v2 benchmark")
    parser.add_argument(
        "--config", type=str, default="benchmarks/configs/flan_v2.yaml",
        help="Benchmark config YAML",
    )
    parser.add_argument(
        "--strategy", type=str, default="similarity",
        help="Strategy name(s), comma-separated (similarity, seqr, classifier, ensemble)",
    )
    parser.add_argument(
        "--regime", type=str, default="non_ood",
        help="Evaluation regime(s), comma-separated (non_ood, semi_ood, ood)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to test data JSON (LoraRetriever format)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic test data (for routing-only eval)",
    )
    parser.add_argument(
        "--samples-per-task", type=int, default=50,
        help="Samples per task for synthetic data",
    )
    parser.add_argument(
        "--adapter-dir", type=str, default=None,
        help="Directory with downloaded adapters",
    )
    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2",
        help="Sentence-transformer encoder name",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top-K adapters to retrieve",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--results-dir", type=str, default="benchmarks/results",
        help="Directory to save JSON results",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate plots (requires matplotlib)",
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    config = FlanV2Config.from_yaml(args.config)
    print(f"  {len(config.tasks)} tasks, {len(config.cluster_metrics)} clusters")

    # Initialize benchmark
    benchmark = FlanV2Benchmark(config)

    # Load test data
    if args.data:
        n = benchmark.load_test_data(args.data)
        print(f"  Loaded {n} test samples from {args.data}")
    elif args.synthetic:
        n = benchmark.load_synthetic_data(samples_per_task=args.samples_per_task)
        print(f"  Generated {n} synthetic samples ({args.samples_per_task}/task)")
    else:
        print("Error: Specify --data or --synthetic")
        sys.exit(1)

    # Build registry
    print(f"\nBuilding registry with encoder: {args.encoder}")
    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(args.encoder)
    except ImportError:
        print("  Warning: sentence-transformers not installed. Using mock embedder.")
        embedder = None

    registry = benchmark.build_registry(
        adapter_dir=args.adapter_dir,
        embedder=embedder,
    )
    print(f"  Registered {registry.size} adapters")

    # Parse strategies and regimes
    strategy_names = [s.strip() for s in args.strategy.split(",")]
    regimes = [r.strip() for r in args.regime.split(",")]

    # Run benchmarks
    all_results: list[BenchmarkResult] = []
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for strategy_name in strategy_names:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        strategy = build_strategy(strategy_name, registry, encoder=args.encoder)

        for regime in regimes:
            print(f"\n  Regime: {regime}")
            start = time.perf_counter()

            result = benchmark.evaluate_routing(
                strategy=strategy,
                registry=registry,
                regime=regime,
                top_k=args.top_k,
            )

            elapsed = time.perf_counter() - start
            all_results.append(result)

            # Print summary
            print(f"  Routing Accuracy (top-1): {result.routing_accuracy_top1:.1%}")
            print(f"  Routing Accuracy (top-3): {result.routing_accuracy_top3:.1%}")
            print(f"  MRR: {result.mrr:.4f}")
            print(f"  NDCG@5: {result.ndcg_at5:.4f}")
            print(f"  Avg routing latency: {result.avg_routing_time_ms:.2f}ms")
            print(f"  Total eval time: {elapsed:.1f}s")

            # Print cluster breakdown
            print("\n  Per-cluster routing accuracy:")
            for cluster, metrics in sorted(result.cluster_results.items()):
                acc = metrics.get("routing_accuracy_top1", 0)
                n = int(metrics.get("n_samples", 0))
                print(f"    {cluster:30s} {acc:.1%} ({n} samples)")

            # Save JSON result
            result_path = results_dir / f"{strategy_name}_{regime}.json"
            benchmark.save_results(result, result_path)
            print(f"\n  Results saved to {result_path}")

    # Generate report
    if args.report:
        print("\nGenerating report...")
        report_content = generate_markdown_report(
            results=[r.to_dict() for r in all_results],
            baselines=config.baselines,
        )
        save_report(report_content, args.report)
        print(f"Report saved to {args.report}")

    # Generate plots
    if args.plots:
        from lora_router.eval.report import generate_plots

        print("\nGenerating plots...")
        plot_paths = generate_plots(
            results=[r.to_dict() for r in all_results],
            output_dir=results_dir / "plots",
            baselines=config.baselines,
        )
        for p in plot_paths:
            print(f"  Plot: {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
