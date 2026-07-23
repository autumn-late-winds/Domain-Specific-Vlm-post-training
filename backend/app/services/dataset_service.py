import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import DEMO_DATASET_ID, DEMO_DPO_FILE, DEMO_TRAIN_FILE, DEMO_VAL_FILE


REQUIRED_SFT_FIELDS = {"id", "image", "task_type", "question", "answer"}
REQUIRED_DPO_FIELDS = {"id", "image", "question", "chosen", "rejected"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def list_datasets() -> list[dict[str, Any]]:
    train = _read_jsonl(DEMO_TRAIN_FILE)
    val = _read_jsonl(DEMO_VAL_FILE)
    return [
        {
            "id": DEMO_DATASET_ID,
            "name": "Synthetic Materials VLM Demo",
            "description": "Tiny synthetic SEM, XRD, Raman, and performance plot examples for mock post-training.",
            "train_examples": len(train),
            "validation_examples": len(val),
            "mode": "demo",
        }
    ]


def get_dataset(dataset_id: str) -> dict[str, Any]:
    if dataset_id != DEMO_DATASET_ID:
        raise KeyError(f"Unknown dataset: {dataset_id}")
    return list_datasets()[0] | {
        "train_file": str(DEMO_TRAIN_FILE),
        "val_file": str(DEMO_VAL_FILE),
        "dpo_file": str(DEMO_DPO_FILE),
    }


def get_examples(dataset_id: str, split: str = "val") -> list[dict[str, Any]]:
    if dataset_id != DEMO_DATASET_ID:
        raise KeyError(f"Unknown dataset: {dataset_id}")
    return _read_jsonl(DEMO_TRAIN_FILE if split == "train" else DEMO_VAL_FILE)


def validate_dataset(dataset_id: str) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    sft_errors = _validate_file(Path(dataset["train_file"]), REQUIRED_SFT_FIELDS)
    val_errors = _validate_file(Path(dataset["val_file"]), REQUIRED_SFT_FIELDS)
    dpo_errors = _validate_file(Path(dataset["dpo_file"]), REQUIRED_DPO_FIELDS)
    errors = sft_errors + val_errors + dpo_errors
    return {
        "dataset_id": dataset_id,
        "valid": len(errors) == 0,
        "errors": errors,
        "checked_files": ["sft_train.jsonl", "sft_val.jsonl", "dpo_train.jsonl"],
    }


def _validate_file(path: Path, fields: set[str]) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(_read_jsonl(path), start=1):
        missing = fields.difference(row)
        if missing:
            errors.append(f"{path.name}:{idx} missing {sorted(missing)}")
    if not path.exists():
        errors.append(f"{path.name} does not exist")
    return errors


def dataset_stats(dataset_id: str) -> dict[str, Any]:
    rows = get_examples(dataset_id, "train") + get_examples(dataset_id, "val")
    by_task = Counter(row.get("task_type", "unknown") for row in rows)
    by_modality = Counter(row.get("metadata", {}).get("modality", "unknown") for row in rows)
    return {
        "dataset_id": dataset_id,
        "total_examples": len(rows),
        "by_task_type": dict(by_task),
        "by_modality": dict(by_modality),
    }

