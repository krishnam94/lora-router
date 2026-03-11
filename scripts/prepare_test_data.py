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

    The dataset has a single default config with a 'task' column.
    We load the full dataset once, then filter per task.

    Returns:
        Number of samples saved.
    """
    from datasets import load_dataset

    all_samples: list[dict[str, str]] = []

    # Load the full dataset once (single default config)
    print("  Loading lorahub/flanv2 dataset...")
    try:
        ds = load_dataset("lorahub/flanv2", split="test")
    except Exception:
        # Try train split if test doesn't exist
        try:
            ds = load_dataset("lorahub/flanv2", split="train")
        except Exception as e:
            print(f"  [FAIL] Could not load lorahub/flanv2: {e}")
            print("  Trying individual task datasets as fallback...")
            return _download_individual_tasks(output_path, samples_per_task)

    # Figure out column names
    columns = ds.column_names
    print(f"  Columns: {columns}")
    print(f"  Total rows: {len(ds)}")

    # Detect task column name
    task_col = None
    for candidate in ["task", "task_name", "dataset", "source"]:
        if candidate in columns:
            task_col = candidate
            break

    if task_col is None:
        print(f"  [WARN] No task column found in {columns}. Using full dataset approach.")
        # If no task column, try to use the data as-is
        return _download_individual_tasks(output_path, samples_per_task)

    # Get unique tasks in the dataset
    unique_tasks = set(ds[task_col])
    print(f"  Unique tasks in dataset: {len(unique_tasks)}")

    # Build a mapping from our task names to dataset task names
    task_name_map: dict[str, str] = {}
    for our_name in FLAN_V2_TASKS:
        if our_name in unique_tasks:
            task_name_map[our_name] = our_name
        else:
            # Try variations
            for ds_name in unique_tasks:
                if ds_name.replace("-", "_") == our_name or ds_name.replace(" ", "_").lower() == our_name:
                    task_name_map[our_name] = ds_name
                    break

    print(f"  Matched {len(task_name_map)}/{len(FLAN_V2_TASKS)} tasks")

    if len(task_name_map) == 0:
        print("  No tasks matched. Falling back to individual datasets...")
        return _download_individual_tasks(output_path, samples_per_task)

    # Detect input/output columns
    input_col = next((c for c in ["inputs", "input", "question", "text"] if c in columns), columns[0])
    target_col = next((c for c in ["targets", "target", "answer", "output", "label"] if c in columns), columns[1] if len(columns) > 1 else columns[0])

    for our_name in FLAN_V2_TASKS:
        ds_name = task_name_map.get(our_name)
        if ds_name is None:
            print(f"  [SKIP] {our_name} (not in dataset)")
            continue

        cluster = CLUSTER_MAP[our_name]
        metric = CLUSTER_METRICS[cluster]

        # Filter dataset for this task
        task_ds = ds.filter(lambda x: x[task_col] == ds_name)
        n = min(samples_per_task, len(task_ds))
        if n == 0:
            print(f"  [SKIP] {our_name} (0 samples)")
            continue

        subset = task_ds.select(range(n))

        for row in subset:
            input_text = str(row.get(input_col, ""))
            target_text = row.get(target_col, "")

            # Handle list-type targets (some datasets return lists)
            if isinstance(target_text, list):
                target_text = target_text[0] if target_text else ""

            all_samples.append({
                "inputs": str(input_text),
                "targets": str(target_text),
                "task": our_name,
                "domain": cluster,
                "metric": metric,
            })

        print(f"  [{n:3d}] {our_name}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    return len(all_samples)


def _download_individual_tasks(
    output_path: Path,
    samples_per_task: int = 50,
) -> int:
    """Fallback: download from individual HF datasets per task.

    Uses the Styxxxx adapter repos which often have training data,
    or well-known HF datasets for each task.
    """
    from datasets import load_dataset

    # Map our task names to known HF dataset IDs
    task_to_hf: dict[str, tuple[str, str | None]] = {
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge"),
        "arc_easy": ("allenai/ai2_arc", "ARC-Easy"),
        "bool_q": ("google/boolq", None),
        "cb": ("aps/super_glue", "cb"),
        "copa": ("aps/super_glue", "copa"),
        "rte": ("aps/super_glue", "rte"),
        "wsc": ("aps/super_glue", "wsc.fixed"),
        "multirc": ("aps/super_glue", "multirc"),
        "record": ("aps/super_glue", "record"),
        "hellaswag": ("Rowan/hellaswag", None),
        "piqa": ("ybisk/piqa", None),
        "sst2": ("stanfordnlp/sst2", None),
        "mnli_matched": ("nyu-mll/glue", "mnli"),
        "mnli_mismatched": ("nyu-mll/glue", "mnli"),
        "qnli": ("nyu-mll/glue", "qnli"),
        "glue_mrpc": ("nyu-mll/glue", "mrpc"),
        "glue_qqp": ("nyu-mll/glue", "qqp"),
        "stsb": ("nyu-mll/glue", "stsb"),
        "wnli": ("nyu-mll/glue", "wnli"),
        "snli": ("stanfordnlp/snli", None),
        "imdb_reviews": ("stanfordnlp/imdb", None),
        "squad_v1": ("rajpurkar/squad", None),
        "squad_v2": ("rajpurkar/squad_v2", None),
        "drop": ("ucinlp/drop", None),
        "openbookqa": ("allenai/openbookqa", None),
        "cosmos_qa": ("cosmos_qa", None),
        "anli_r1": ("facebook/anli", None),
        "anli_r2": ("facebook/anli", None),
        "anli_r3": ("facebook/anli", None),
        "common_gen": ("allenai/common_gen", None),
        "trivia_qa": ("mandarjoshi/trivia_qa", "rc"),
        "natural_questions": ("google-research-datasets/natural_questions", None),
    }

    all_samples: list[dict[str, str]] = []
    loaded = 0

    for task_name in FLAN_V2_TASKS:
        cluster = CLUSTER_MAP[task_name]
        metric = CLUSTER_METRICS[cluster]

        hf_info = task_to_hf.get(task_name)
        if hf_info is None:
            print(f"  [SKIP] {task_name} (no HF mapping)")
            continue

        dataset_id, config = hf_info

        try:
            if config:
                ds = load_dataset(dataset_id, config, split="validation")
            else:
                # Try validation first, then test
                try:
                    ds = load_dataset(dataset_id, split="validation")
                except Exception:
                    ds = load_dataset(dataset_id, split="test")
        except Exception as e:
            print(f"  [SKIP] {task_name}: {e}")
            continue

        n = min(samples_per_task, len(ds))
        subset = ds.select(range(n))

        # Generic column extraction
        cols = subset.column_names
        for row in subset:
            # Try common input column names
            input_text = ""
            for col in ["question", "premise", "sentence", "sentence1", "text", "passage"]:
                if col in cols and row[col]:
                    input_text = str(row[col])
                    break
            if not input_text and cols:
                input_text = str(row[cols[0]])

            # Try common target column names
            target_text = ""
            for col in ["answer", "label", "answers", "target", "hypothesis"]:
                if col in cols and row[col] is not None:
                    val = row[col]
                    if isinstance(val, dict) and "text" in val:
                        target_text = str(val["text"][0]) if val["text"] else ""
                    elif isinstance(val, list):
                        target_text = str(val[0]) if val else ""
                    else:
                        target_text = str(val)
                    break

            all_samples.append({
                "inputs": input_text,
                "targets": target_text,
                "task": task_name,
                "domain": cluster,
                "metric": metric,
            })

        loaded += 1
        print(f"  [{n:3d}] {task_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\n  Loaded {loaded}/{len(FLAN_V2_TASKS)} tasks via individual datasets")
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
