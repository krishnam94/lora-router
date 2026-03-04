"""Report generation for benchmark results.

Generates markdown tables and comparison reports. Optionally generates
matplotlib plots for visual comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_markdown_report(
    results: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]] | None = None,
    title: str = "FLAN v2 Benchmark Results",
) -> str:
    """Generate a markdown report from benchmark results.

    Args:
        results: List of BenchmarkResult.to_dict() outputs.
        baselines: Optional baseline scores for comparison.
        title: Report title.

    Returns:
        Markdown string.
    """
    lines = [f"# {title}", ""]

    for result in results:
        regime = result.get("regime", "unknown")
        strategy = result.get("strategy", "unknown")
        lines.append(f"## {strategy} - {regime.upper()}")
        lines.append("")

        # Summary table
        lines.append("### Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Routing Accuracy (top-1) | {result.get('routing_accuracy_top1', 0):.1%} |")
        lines.append(f"| Routing Accuracy (top-3) | {result.get('routing_accuracy_top3', 0):.1%} |")
        lines.append(f"| MRR | {result.get('mrr', 0):.4f} |")
        lines.append(f"| NDCG@5 | {result.get('ndcg_at5', 0):.4f} |")
        if result.get("normalized_oracle") is not None:
            lines.append(f"| Normalized Oracle Score | {result['normalized_oracle']:.1f}% |")
        lines.append(f"| Avg Routing Latency | {result.get('avg_routing_time_ms', 0):.2f}ms |")
        lines.append(f"| Total Samples | {result.get('n_samples', 0)} |")
        lines.append("")

        # Cluster breakdown
        cluster_results = result.get("cluster_results", {})
        if cluster_results:
            lines.append("### Per-Cluster Results")
            lines.append("")
            lines.append("| Cluster | Routing Acc | Samples |")
            lines.append("|---------|------------|---------|")
            for cluster, metrics in sorted(cluster_results.items()):
                acc = metrics.get("routing_accuracy_top1", 0)
                n = int(metrics.get("n_samples", 0))
                lines.append(f"| {cluster} | {acc:.1%} | {n} |")
            lines.append("")

        # Per-task results
        task_results = result.get("task_results", {})
        if task_results:
            lines.append("### Per-Task Results")
            lines.append("")
            lines.append("| Task | Cluster | Acc (top-1) | Acc (top-3) | MRR | Samples |")
            lines.append("|------|---------|-------------|-------------|-----|---------|")
            for task_name, tr in sorted(task_results.items()):
                lines.append(
                    f"| {task_name} | {tr.get('cluster', '')} | "
                    f"{tr.get('accuracy_top1', 0):.1%} | "
                    f"{tr.get('accuracy_top3', 0):.1%} | "
                    f"{tr.get('mrr', 0):.3f} | "
                    f"{tr.get('n_samples', 0)} |"
                )
            lines.append("")

    # Baseline comparison
    if baselines and results:
        lines.append("## Comparison to Baselines")
        lines.append("")
        lines.append("| Method | Non-OOD | OOD | Source |")
        lines.append("|--------|---------|-----|--------|")

        for name, scores in sorted(baselines.items()):
            non_ood = scores.get("non_ood", "-")
            ood = scores.get("ood", "-")
            source = scores.get("source", "")
            non_ood_str = f"{non_ood}%" if isinstance(non_ood, (int, float)) else str(non_ood)
            ood_str = f"{ood}%" if isinstance(ood, (int, float)) else str(ood)
            lines.append(f"| {name} | {non_ood_str} | {ood_str} | {source} |")

        # Add our results
        for result in results:
            regime = result.get("regime", "")
            norm = result.get("normalized_oracle")
            if norm is not None:
                strategy = result.get("strategy", "ours")
                lines.append(f"| **{strategy} ({regime})** | {norm:.1f}% | - | This work |")
        lines.append("")

    return "\n".join(lines)


def generate_comparison_table(
    results: list[dict[str, Any]],
    metric: str = "routing_accuracy_top1",
) -> str:
    """Generate a side-by-side comparison table for multiple strategies.

    Args:
        results: List of BenchmarkResult dicts from different strategies.
        metric: Which metric to compare.

    Returns:
        Markdown table string.
    """
    if not results:
        return ""

    lines = ["| Strategy | Regime | " + metric + " |"]
    lines.append("|----------|--------|" + "-" * (len(metric) + 2) + "|")

    for result in sorted(results, key=lambda r: (r.get("regime", ""), r.get("strategy", ""))):
        strategy = result.get("strategy", "unknown")
        regime = result.get("regime", "")
        value = result.get(metric, 0)
        if isinstance(value, float) and value <= 1.0:
            lines.append(f"| {strategy} | {regime} | {value:.1%} |")
        else:
            lines.append(f"| {strategy} | {regime} | {value} |")

    return "\n".join(lines)


def save_report(
    content: str,
    path: str | Path,
) -> None:
    """Save report content to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def generate_plots(
    results: list[dict[str, Any]],
    output_dir: str | Path,
    baselines: dict[str, dict[str, float]] | None = None,
) -> list[Path]:
    """Generate matplotlib plots for benchmark results.

    Creates:
    1. Bar chart comparing routing accuracy across strategies
    2. Per-cluster heatmap
    3. Comparison to baselines (if provided)

    Args:
        results: List of BenchmarkResult dicts.
        output_dir: Directory to save plots.
        baselines: Optional baseline scores.

    Returns:
        List of generated plot file paths.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []

    # Plot 1: Routing accuracy comparison
    if results:
        fig, ax = plt.subplots(figsize=(10, 6))
        strategies = [r.get("strategy", "") for r in results]
        acc_top1 = [r.get("routing_accuracy_top1", 0) * 100 for r in results]
        acc_top3 = [r.get("routing_accuracy_top3", 0) * 100 for r in results]

        x = range(len(strategies))
        width = 0.35
        ax.bar([i - width / 2 for i in x], acc_top1, width, label="Top-1", color="#2563eb")
        ax.bar([i + width / 2 for i in x], acc_top3, width, label="Top-3", color="#7c3aed")
        ax.set_ylabel("Routing Accuracy (%)")
        ax.set_title("Routing Accuracy by Strategy")
        ax.set_xticks(list(x))
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 105)
        fig.tight_layout()

        path = output_dir / "routing_accuracy.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    # Plot 2: Per-cluster breakdown
    for result in results:
        cluster_results = result.get("cluster_results", {})
        if not cluster_results:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        clusters = sorted(cluster_results.keys())
        accs = [cluster_results[c].get("routing_accuracy_top1", 0) * 100 for c in clusters]

        bars = ax.barh(clusters, accs, color="#2563eb")
        ax.set_xlabel("Routing Accuracy (%)")
        ax.set_title(f"Per-Cluster Accuracy - {result.get('strategy', '')} ({result.get('regime', '')})")
        ax.set_xlim(0, 105)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.1f}%", va="center", fontsize=9)

        fig.tight_layout()
        strategy = result.get("strategy", "unknown").lower().replace(" ", "_")
        regime = result.get("regime", "")
        path = output_dir / f"cluster_{strategy}_{regime}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    # Plot 3: Baseline comparison
    if baselines and any(r.get("normalized_oracle") for r in results):
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(baselines.keys())
        ood_scores = [baselines[m].get("ood", 0) for m in methods]

        # Add our results
        for result in results:
            if result.get("normalized_oracle") and result.get("regime") == "ood":
                methods.append(f"Ours ({result.get('strategy', '')})")
                ood_scores.append(result["normalized_oracle"])

        colors = ["#94a3b8"] * len(baselines) + ["#2563eb"] * (len(methods) - len(baselines))
        ax.barh(methods, ood_scores, color=colors)
        ax.set_xlabel("Normalized Oracle Score (%)")
        ax.set_title("OOD Performance vs Baselines")
        ax.axvline(x=100, color="red", linestyle="--", alpha=0.5, label="Oracle")

        for i, score in enumerate(ood_scores):
            if score > 0:
                ax.text(score + 0.5, i, f"{score:.1f}%", va="center", fontsize=9)

        fig.tight_layout()
        path = output_dir / "baseline_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    return plots
