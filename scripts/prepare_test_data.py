"""Download and prepare FLAN v2 test data for benchmark evaluation.

Uses HuggingFace datasets in STREAMING mode to avoid downloading
full datasets to disk (some are 50GB+). Only fetches the samples we need.

Usage:
    python scripts/prepare_test_data.py --output benchmarks/data/combined_test.json
    python scripts/prepare_test_data.py --output benchmarks/data/combined_test.json --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
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

# Task -> (hf_dataset_id, config, split, input_cols, target_cols)
# Uses streaming mode - no data downloaded to disk.
TASK_TO_HF: dict[str, tuple[str, str | None, str, list[str], list[str]]] = {
    # Closed-book QA
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation", ["question"], ["answerKey"]),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "validation", ["question"], ["answerKey"]),
    "natural_questions": ("google-research-datasets/nq_open", None, "validation", ["question"], ["answer"]),
    "trivia_qa": ("mandarjoshi/trivia_qa", "rc", "validation", ["question"], ["answer"]),
    # Commonsense
    "copa": ("aps/super_glue", "copa", "validation", ["premise"], ["label"]),
    "hellaswag": ("Rowan/hellaswag", None, "validation", ["ctx"], ["label"]),
    "piqa": ("ybisk/piqa", None, "validation", ["goal"], ["label"]),
    # Coreference
    "wsc": ("aps/super_glue", "wsc.fixed", "validation", ["text"], ["label"]),
    # NLI
    "anli_r1": ("facebook/anli", None, "test_r1", ["premise", "hypothesis"], ["label"]),
    "anli_r2": ("facebook/anli", None, "test_r2", ["premise", "hypothesis"], ["label"]),
    "anli_r3": ("facebook/anli", None, "test_r3", ["premise", "hypothesis"], ["label"]),
    "cb": ("aps/super_glue", "cb", "validation", ["premise", "hypothesis"], ["label"]),
    "mnli_matched": ("nyu-mll/glue", "mnli", "validation_matched", ["premise", "hypothesis"], ["label"]),
    "mnli_mismatched": ("nyu-mll/glue", "mnli", "validation_mismatched", ["premise", "hypothesis"], ["label"]),
    "qnli": ("nyu-mll/glue", "qnli", "validation", ["question", "sentence"], ["label"]),
    "rte": ("aps/super_glue", "rte", "validation", ["premise", "hypothesis"], ["label"]),
    "snli": ("stanfordnlp/snli", None, "validation", ["premise", "hypothesis"], ["label"]),
    "wnli": ("nyu-mll/glue", "wnli", "validation", ["sentence1", "sentence2"], ["label"]),
    # Paraphrase
    "glue_mrpc": ("nyu-mll/glue", "mrpc", "validation", ["sentence1", "sentence2"], ["label"]),
    "glue_qqp": ("nyu-mll/glue", "qqp", "validation", ["question1", "question2"], ["label"]),
    "paws_wiki": ("paws", "labeled_final", "validation", ["sentence1", "sentence2"], ["label"]),
    "stsb": ("nyu-mll/glue", "stsb", "validation", ["sentence1", "sentence2"], ["label"]),
    # Reading comprehension (commonsense)
    "cosmos_qa": ("cosmos_qa", None, "validation", ["context", "question"], ["label"]),
    "record": ("aps/super_glue", "record", "validation", ["passage", "query"], ["answers"]),
    # Reading comprehension
    "bool_q": ("google/boolq", None, "validation", ["question", "passage"], ["answer"]),
    "drop": ("ucinlp/drop", None, "validation", ["passage", "question"], ["answers_spans"]),
    "multirc": ("aps/super_glue", "multirc", "validation", ["paragraph", "question"], ["label"]),
    "openbookqa": ("allenai/openbookqa", "main", "validation", ["question_stem"], ["answerKey"]),
    "squad_v1": ("rajpurkar/squad", None, "validation", ["question", "context"], ["answers"]),
    "squad_v2": ("rajpurkar/squad_v2", None, "validation", ["question", "context"], ["answers"]),
    # Sentiment
    "imdb_reviews": ("stanfordnlp/imdb", None, "test", ["text"], ["label"]),
    "sst2": ("stanfordnlp/sst2", None, "validation", ["sentence"], ["label"]),
    "yelp_polarity_reviews": ("fancyzhx/yelp_polarity", None, "test", ["text"], ["label"]),
    "sentiment140": ("stanfordnlp/sentiment140", None, "test", ["text"], ["sentiment"]),
    # Struct to text
    "common_gen": ("allenai/common_gen", None, "validation", ["concepts"], ["target"]),
    "dart": ("GEM/dart", None, "validation", ["tripleset"], ["target"]),
    "e2e_nlg": ("GEM/e2e_nlg", None, "validation", ["meaning_representation"], ["target"]),
    "web_nlg_en": ("GEM/web_nlg", "en", "validation", ["input"], ["target"]),
    # Translation
    "para_crawl_enes": ("para_crawl", "enes", "train", ["translation"], []),
    "wmt14_enfr": ("wmt/wmt14", "fr-en", "validation", ["translation"], []),
    "wmt16_translate_csen": ("wmt/wmt16", "cs-en", "validation", ["translation"], []),
    "wmt16_translate_deen": ("wmt/wmt16", "de-en", "validation", ["translation"], []),
    "wmt16_translate_fien": ("wmt/wmt16", "fi-en", "validation", ["translation"], []),
    "wmt16_translate_roen": ("wmt/wmt16", "ro-en", "validation", ["translation"], []),
    "wmt16_translate_ruen": ("wmt/wmt16", "ru-en", "validation", ["translation"], []),
    "wmt16_translate_tren": ("wmt/wmt16", "tr-en", "validation", ["translation"], []),
}

# Label maps for classification tasks
LABEL_MAPS = {
    "anli_r1": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "anli_r2": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "anli_r3": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "cb": {0: "entailment", 1: "contradiction", 2: "neutral"},
    "mnli_matched": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "mnli_mismatched": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "snli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "wnli": {0: "not_entailment", 1: "entailment"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "glue_mrpc": {0: "not_equivalent", 1: "equivalent"},
    "glue_qqp": {0: "not_duplicate", 1: "duplicate"},
    "paws_wiki": {0: "not_paraphrase", 1: "paraphrase"},
    "stsb": None,  # Regression - use raw score
    "copa": {0: "choice1", 1: "choice2"},
    "hellaswag": None,  # Index label
    "piqa": {0: "solution1", 1: "solution2"},
    "wsc": {0: "False", 1: "True"},
    "cosmos_qa": {0: "A", 1: "B", 2: "C", 3: "D"},
    "multirc": {0: "False", 1: "True"},
    "bool_q": {False: "false", True: "true"},
    "imdb_reviews": {0: "negative", 1: "positive"},
    "sst2": {0: "negative", 1: "positive"},
    "yelp_polarity_reviews": {0: "negative", 1: "positive"},
    "sentiment140": {0: "negative", 4: "positive"},
}


def _extract_target(row: dict, task_name: str, target_cols: list[str]) -> str:
    """Extract target text from a row, handling various formats."""
    label_map = LABEL_MAPS.get(task_name)

    for col in target_cols:
        val = row.get(col)
        if val is None:
            continue

        # Translation pairs
        if isinstance(val, dict) and "en" in val:
            return str(val.get("en", ""))

        # SQuAD-style answers
        if isinstance(val, dict) and "text" in val:
            texts = val["text"]
            return str(texts[0]) if texts else ""

        # DROP-style answers_spans
        if isinstance(val, dict) and "spans" in val:
            spans = val["spans"]
            return str(spans[0]) if spans else ""

        # ReCoRD-style answers (list of strings)
        if isinstance(val, list) and val:
            if isinstance(val[0], str):
                return val[0]
            return str(val[0])

        # Integer labels with mapping
        if isinstance(val, (int, bool)) and label_map is not None:
            return str(label_map.get(val, str(val)))

        # Float (stsb regression)
        if isinstance(val, float):
            return str(round(val, 1))

        return str(val)

    return ""


def _extract_input(row: dict, input_cols: list[str]) -> str:
    """Extract input text from a row."""
    parts = []
    for col in input_cols:
        val = row.get(col)
        if val is None:
            continue

        # Translation pairs - use source language
        if isinstance(val, dict):
            # For translation, pick non-English key as source
            for k, v in val.items():
                if k != "en":
                    parts.append(str(v))
                    break
            else:
                # If only English, just use it
                parts.append(str(list(val.values())[0]))
            continue

        # Lists (common_gen concepts)
        if isinstance(val, list):
            parts.append(", ".join(str(x) for x in val))
            continue

        parts.append(str(val))

    return " ".join(parts) if parts else ""


def download_streaming(
    output_path: Path,
    samples_per_task: int = 50,
) -> int:
    """Download test data using HF streaming mode - no disk usage.

    Streams samples over HTTP without downloading full datasets.
    """
    from datasets import load_dataset

    all_samples: list[dict[str, str]] = []
    loaded = 0
    skipped = []

    for task_name in FLAN_V2_TASKS:
        cluster = CLUSTER_MAP[task_name]
        metric = CLUSTER_METRICS[cluster]

        hf_info = TASK_TO_HF.get(task_name)
        if hf_info is None:
            skipped.append(task_name)
            print(f"  [SKIP] {task_name} (no HF mapping)")
            continue

        dataset_id, config, split, input_cols, target_cols = hf_info

        try:
            kwargs: dict = {"streaming": True, "split": split}
            if config:
                ds_iter = load_dataset(dataset_id, config, **kwargs)
            else:
                ds_iter = load_dataset(dataset_id, **kwargs)

            # Take only the samples we need via streaming
            count = 0
            for row in islice(ds_iter, samples_per_task):
                input_text = _extract_input(row, input_cols)
                target_text = _extract_target(row, task_name, target_cols)

                # For translation tasks with no target_cols, extract target from translation dict
                if not target_cols and "translation" in row:
                    target_text = str(row["translation"].get("en", ""))

                if not input_text:
                    continue

                all_samples.append({
                    "inputs": input_text,
                    "targets": target_text,
                    "task": task_name,
                    "domain": cluster,
                    "metric": metric,
                })
                count += 1

            if count > 0:
                loaded += 1
                print(f"  [{count:3d}] {task_name}")
            else:
                skipped.append(task_name)
                print(f"  [SKIP] {task_name} (0 samples extracted)")

        except Exception as e:
            skipped.append(task_name)
            err_msg = str(e)[:100]
            print(f"  [SKIP] {task_name}: {err_msg}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\n  Loaded {loaded}/{len(FLAN_V2_TASKS)} tasks ({len(all_samples)} samples)")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")

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

    # Download using streaming (no disk usage)
    print(f"Downloading FLAN v2 test data ({args.samples_per_task} samples/task)")
    print("Mode: streaming (no disk cache)\n")

    n = download_streaming(output_path, samples_per_task=args.samples_per_task)

    print(f"\nDone: {n} samples saved to {output_path}")

    # Verify
    counts = verify_test_data(output_path)
    covered = sum(1 for t in FLAN_V2_TASKS if t in counts)
    print(f"Coverage: {covered}/{len(FLAN_V2_TASKS)} tasks")


if __name__ == "__main__":
    main()
