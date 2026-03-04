"""Download all 48 Styxxxx FLAN v2 LoRA adapters from HuggingFace.

Usage:
    python scripts/download_flan_adapters.py --output-dir adapters/flan_v2
    python scripts/download_flan_adapters.py --output-dir adapters/flan_v2 --tasks arc_challenge,cb
    python scripts/download_flan_adapters.py --output-dir adapters/flan_v2 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# All 48 FLAN v2 benchmark tasks
FLAN_V2_TASKS = [
    # Closed-book QA (4)
    "arc_challenge", "arc_easy", "natural_questions", "trivia_qa",
    # Commonsense (4)
    "copa", "hellaswag", "piqa", "story_cloze",
    # Coreference (2)
    "definite_pronoun_resolution", "wsc",
    # NLI (10)
    "anli_r1", "anli_r2", "anli_r3", "cb", "mnli_matched",
    "mnli_mismatched", "qnli", "rte", "snli", "wnli",
    # Paraphrase (4)
    "glue_mrpc", "glue_qqp", "paws_wiki", "stsb",
    # Reading Comp + Commonsense (2)
    "cosmos_qa", "record",
    # Reading Comprehension (6)
    "bool_q", "drop", "multirc", "openbookqa", "squad_v1", "squad_v2",
    # Sentiment (4)
    "imdb_reviews", "sentiment140", "sst2", "yelp_polarity_reviews",
    # Struct-to-Text (4)
    "common_gen", "dart", "e2e_nlg", "web_nlg_en",
    # Translation (8)
    "para_crawl_enes", "wmt14_enfr", "wmt16_translate_csen",
    "wmt16_translate_deen", "wmt16_translate_fien", "wmt16_translate_roen",
    "wmt16_translate_ruen", "wmt16_translate_tren",
]

ADAPTER_PREFIX = "Styxxxx/llama2_7b_lora"


def hf_id(task: str) -> str:
    return f"{ADAPTER_PREFIX}-{task}"


def download_adapter(task: str, output_dir: Path, force: bool = False) -> bool:
    """Download a single adapter from HuggingFace Hub.

    Returns True if successful, False otherwise.
    """
    from huggingface_hub import snapshot_download

    adapter_id = hf_id(task)
    local_dir = output_dir / task

    if local_dir.exists() and not force:
        adapter_config = local_dir / "adapter_config.json"
        if adapter_config.exists():
            print(f"  [skip] {task} - already downloaded")
            return True

    try:
        snapshot_download(
            repo_id=adapter_id,
            local_dir=str(local_dir),
            repo_type="model",
        )
        print(f"  [done] {task}")
        return True
    except Exception as e:
        print(f"  [FAIL] {task} - {e}")
        return False


def verify_adapter(task: str, output_dir: Path) -> bool:
    """Check that an adapter was downloaded correctly."""
    local_dir = output_dir / task
    required_files = ["adapter_config.json"]
    weight_files = ["adapter_model.safetensors", "adapter_model.bin"]

    for f in required_files:
        if not (local_dir / f).exists():
            return False

    # At least one weight file must exist
    return any((local_dir / f).exists() for f in weight_files)


def generate_registry_yaml(output_dir: Path, tasks: list[str]) -> Path:
    """Generate a YAML registry file for downloaded adapters."""
    adapters = []
    for task in tasks:
        if verify_adapter(task, output_dir):
            adapters.append({
                "name": task,
                "path": str(output_dir / task),
                "description": f"FLAN v2 LoRA adapter for {task.replace('_', ' ')}",
                "domain": _get_cluster(task),
            })

    yaml_path = output_dir / "registry.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(adapters, f, default_flow_style=False, sort_keys=False)
    return yaml_path


# Cluster lookup
_CLUSTER_MAP = {
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


def _get_cluster(task: str) -> str:
    return _CLUSTER_MAP.get(task, "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FLAN v2 LoRA adapters")
    parser.add_argument(
        "--output-dir", type=str, default="adapters/flan_v2",
        help="Directory to save adapters",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task names (default: all 48)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing adapters")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing downloads")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tasks = args.tasks.split(",") if args.tasks else FLAN_V2_TASKS

    # Validate task names
    invalid = [t for t in tasks if t not in FLAN_V2_TASKS]
    if invalid:
        print(f"Error: Unknown tasks: {invalid}")
        print(f"Valid tasks: {FLAN_V2_TASKS}")
        sys.exit(1)

    if args.dry_run:
        print(f"Would download {len(tasks)} adapters to {output_dir}/")
        for task in tasks:
            print(f"  {hf_id(task)} -> {output_dir / task}")
        return

    if args.verify_only:
        print(f"Verifying {len(tasks)} adapters in {output_dir}/")
        ok = sum(1 for t in tasks if verify_adapter(t, output_dir))
        missing = len(tasks) - ok
        print(f"\nResult: {ok}/{len(tasks)} verified, {missing} missing")
        if missing > 0:
            for t in tasks:
                if not verify_adapter(t, output_dir):
                    print(f"  [missing] {t}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(tasks)} adapters to {output_dir}/\n")

    success = 0
    failed = []
    for task in tasks:
        if download_adapter(task, output_dir, force=args.force):
            success += 1
        else:
            failed.append(task)

    print(f"\nDone: {success}/{len(tasks)} downloaded")
    if failed:
        print(f"Failed: {failed}")

    # Generate registry YAML
    yaml_path = generate_registry_yaml(output_dir, tasks)
    print(f"Registry written to {yaml_path}")

    # Final verification
    verified = sum(1 for t in tasks if verify_adapter(t, output_dir))
    print(f"Verified: {verified}/{len(tasks)}")


if __name__ == "__main__":
    main()
