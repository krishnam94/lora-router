"""Download and prepare FLAN v2 test data for benchmark evaluation.

Sources (tried in order):
1. LoraRetriever's combined_test.json from GitHub
2. HuggingFace lorahub/flanv2 dataset (sample and format)

Usage:
    python scripts/prepare_test_data.py --output benchmarks/data/combined_test.json
    python scripts/prepare_test_data.py --output benchmarks/data/combined_test.json --source hf
    python scripts/prepare_test_data.py --output benchmarks/data/combined_test.json --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# All 48 FLAN v2 benchmark tasks
FLAN_V2_TASKS = [
    "arc_challenge", "arc_easy", "natural_questions", "trivia_qa",
    "copa", "hellaswag", "piqa", "story_cloze",
    "definite_pronoun_resolution", "wsc",
    "anli_r1", "anli_r2", "anli_r3", "cb", "mnli_matched",
    "mnli_mismatched", "qnli", "rte", "snli", "wnli",
    "glue_mrpc", "glue_qqp", "paws_wiki", "stsb",
    "cosmos_qa", "record",
    "bool_q", "drop", "multirc", "openbookqa", "squad_v1", "squad_v2",
    "imdb_reviews", "sentiment140", "sst2", "yelp_polarity_reviews",
    "common_gen", "dart", "e2e_nlg", "web_nlg_en",
    "para_crawl_enes", "wmt14_enfr", "wmt16_translate_csen",
    "wmt16_translate_deen", "wmt16_translate_fien", "wmt16_translate_roen",
    "wmt16_translate_ruen", "wmt16_translate_tren",
]

# Task -> cluster mapping
CLUSTER_MAP = {
    "arc_challenge": "closed_book_qa", "arc_easy": "closed_book_qa",
    "natural_questions": "closed_book_qa", "trivia_qa": "closed_book_qa",
    "copa": "commonsense", "hellaswag": "commonsense",
    "piqa": "commonsense", "story_cloze": "commonsense",
    "definite_pronoun_resolution": "coreference", "wsc": "coreference",
    "anli_r1": "nli", "anli_r2": "nli", "anli_r3": "nli", "cb": "nli",
    "mnli_matched": "nli", "mnli_mismatched": "nli", "qnli": "nli",
    "rte": "nli", "snli": "nli", "wnli": "nli",
    "glue_mrpc": "paraphrase", "glue_qqp": "paraphrase",
    "paws_wiki": "paraphrase", "stsb": "paraphrase",
    "cosmos_qa": "reading_comp_commonsense", "record": "reading_comp_commonsense",
    "bool_q": "reading_comprehension", "drop": "reading_comprehension",
    "multirc": "reading_comprehension", "openbookqa": "reading_comprehension",
    "squad_v1": "reading_comprehension", "squad_v2": "reading_comprehension",
    "imdb_reviews": "sentiment", "sentiment140": "sentiment",
    "sst2": "sentiment", "yelp_polarity_reviews": "sentiment",
    "common_gen": "struct_to_text", "dart": "struct_to_text",
    "e2e_nlg": "struct_to_text", "web_nlg_en": "struct_to_text",
    "para_crawl_enes": "translation", "wmt14_enfr": "translation",
    "wmt16_translate_csen": "translation", "wmt16_translate_deen": "translation",
    "wmt16_translate_fien": "translation", "wmt16_translate_roen": "translation",
    "wmt16_translate_ruen": "translation", "wmt16_translate_tren": "translation",
}

# Cluster -> metric mapping
CLUSTER_METRICS = {
    "closed_book_qa": "exact_match",
    "commonsense": "exact_match",
    "coreference": "exact_match",
    "nli": "exact_match",
    "paraphrase": "exact_match",
    "reading_comp_commonsense": "exact_match",
    "reading_comprehension": "exact_match",
    "sentiment": "exact_match",
    "struct_to_text": "rouge",
    "translation": "bleu",
}


def download_from_hf(
    output_path: Path,
    samples_per_task: int = 50,
) -> int:
    """Download test data from HuggingFace lorahub/flanv2 dataset.

    Samples from each task's test split and formats into LoraRetriever format.

    Returns:
        Number of samples saved.
    """
    from datasets import load_dataset

    all_samples: list[dict[str, str]] = []

    for task_name in FLAN_V2_TASKS:
        cluster = CLUSTER_MAP[task_name]
        metric = CLUSTER_METRICS[cluster]

        try:
            # lorahub/flanv2 is organized by task name
            ds = load_dataset("lorahub/flanv2", task_name, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  [WARN] Could not load {task_name}: {e}")
            # Try alternate naming (underscores to hyphens, etc.)
            try:
                alt_name = task_name.replace("_", "-")
                ds = load_dataset("lorahub/flanv2", alt_name, split="test", trust_remote_code=True)
            except Exception:
                print(f"  [SKIP] {task_name}")
                continue

        # Sample up to samples_per_task
        n = min(samples_per_task, len(ds))
        indices = list(range(n))
        subset = ds.select(indices)

        for row in subset:
            # HF dataset typically has 'inputs' and 'targets' columns
            input_text = row.get("inputs", row.get("input", row.get("question", "")))
            target_text = row.get("targets", row.get("target", row.get("answer", "")))

            # Handle list-type targets (some datasets return lists)
            if isinstance(target_text, list):
                target_text = target_text[0] if target_text else ""

            all_samples.append({
                "inputs": str(input_text),
                "targets": str(target_text),
                "task": task_name,
                "domain": cluster,
                "metric": metric,
            })

        print(f"  [{n:3d}] {task_name}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    return len(all_samples)


def verify_test_data(path: Path) -> dict[str, int]:
    """Verify test data file and return per-task sample counts."""
    if not path.exists():
        print(f"File not found: {path}")
        return {}

    with open(path) as f:
        data = json.load(f)

    task_counts: dict[str, int] = {}
    for entry in data:
        task = entry.get("task", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1

    return task_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FLAN v2 test data")
    parser.add_argument(
        "--output", type=str, default="benchmarks/data/combined_test.json",
        help="Output path for test data JSON",
    )
    parser.add_argument(
        "--source", type=str, default="hf",
        choices=["hf"],
        help="Data source: hf (HuggingFace lorahub/flanv2)",
    )
    parser.add_argument(
        "--samples-per-task", type=int, default=50,
        help="Number of test samples per task (default: 50)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify existing test data",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.verify_only:
        print(f"Verifying test data at {output_path}")
        counts = verify_test_data(output_path)
        if not counts:
            sys.exit(1)

        total = sum(counts.values())
        n_tasks = len(counts)
        missing = set(FLAN_V2_TASKS) - set(counts.keys())

        print(f"\nTotal: {total} samples across {n_tasks} tasks")
        print(f"Expected: {len(FLAN_V2_TASKS)} tasks")

        if missing:
            print(f"\nMissing tasks ({len(missing)}):")
            for t in sorted(missing):
                print(f"  - {t}")
        else:
            print("\nAll 48 tasks present.")

        # Per-cluster summary
        cluster_counts: dict[str, int] = {}
        for task, count in counts.items():
            cluster = CLUSTER_MAP.get(task, "unknown")
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + count

        print("\nPer-cluster:")
        for cluster, count in sorted(cluster_counts.items()):
            print(f"  {cluster:30s} {count:5d} samples")

        return

    # Download
    print(f"Downloading FLAN v2 test data ({args.samples_per_task} samples/task)")
    print(f"Source: {args.source}")
    print(f"Output: {output_path}\n")

    n = download_from_hf(output_path, samples_per_task=args.samples_per_task)

    print(f"\nDone: {n} samples saved to {output_path}")

    # Verify
    counts = verify_test_data(output_path)
    covered = sum(1 for t in FLAN_V2_TASKS if t in counts)
    print(f"Coverage: {covered}/{len(FLAN_V2_TASKS)} tasks")


if __name__ == "__main__":
    main()
