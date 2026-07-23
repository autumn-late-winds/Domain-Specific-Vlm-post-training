import json
import re
from collections import defaultdict
from datetime import datetime, timezone

from app.config import OUTPUT_DIR
from app.services.dataset_service import get_examples
from app.services.mock_vlm import MockVLM
from app.services.training_service import get_experiment


DOMAIN_KEYWORDS = {
    "porous",
    "nanosheet",
    "hierarchical",
    "xrd",
    "raman",
    "peak",
    "sample",
    "performance",
    "spectrum",
    "capacity",
    "morphology",
    "grounded",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    common = set(pred).intersection(ref)
    precision = len(common) / len(set(pred))
    recall = len(common) / len(set(ref))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def keyword_coverage(text: str) -> float:
    tokens = set(_tokens(text))
    hits = tokens.intersection(DOMAIN_KEYWORDS)
    return round(len(hits) / 5, 4) if hits else 0.0


def evaluate_experiment(experiment_id: str, dataset_id: str) -> dict:
    experiment = get_experiment(experiment_id)
    rows = get_examples(dataset_id, "val")
    base = _evaluate_rows(rows, MockVLM("base"))
    finetuned = _evaluate_rows(rows, MockVLM("finetuned"))
    report = _compare(base, finetuned, rows, experiment)
    out_dir = OUTPUT_DIR / experiment_id
    (out_dir / "evaluation_base.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    (out_dir / "evaluation_finetuned.json").write_text(json.dumps(finetuned, indent=2), encoding="utf-8")
    (out_dir / "comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _evaluate_rows(rows: list[dict], model: MockVLM) -> dict:
    scored = []
    per_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        prediction = model.generate(row["question"], row["task_type"], row.get("metadata", {}))
        score = {
            "id": row["id"],
            "task_type": row["task_type"],
            "question": row["question"],
            "reference": row["answer"],
            "prediction": prediction,
            "token_f1": token_f1(prediction, row["answer"]),
            "keyword_coverage": keyword_coverage(prediction),
            "length_sanity": 1.0 if 8 <= len(_tokens(prediction)) <= 80 else 0.0,
        }
        score["overall"] = round((score["token_f1"] + score["keyword_coverage"] + score["length_sanity"]) / 3, 4)
        scored.append(score)
        per_task[row["task_type"]].append(score)
    return {"aggregate": _aggregate(scored), "per_task": {k: _aggregate(v) for k, v in per_task.items()}, "examples": scored}


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "overall": 0, "token_f1": 0, "keyword_coverage": 0, "length_sanity": 0}
    keys = ["overall", "token_f1", "keyword_coverage", "length_sanity"]
    return {"count": len(rows)} | {key: round(sum(row[key] for row in rows) / len(rows), 4) for key in keys}


def _compare(base: dict, finetuned: dict, rows: list[dict], experiment: dict) -> dict:
    before_after = []
    for base_row, tuned_row in zip(base["examples"], finetuned["examples"]):
        before_after.append(
            {
                "id": base_row["id"],
                "task_type": base_row["task_type"],
                "question": base_row["question"],
                "reference": base_row["reference"],
                "base_prediction": base_row["prediction"],
                "finetuned_prediction": tuned_row["prediction"],
                "delta_overall": round(tuned_row["overall"] - base_row["overall"], 4),
            }
        )
    failures = [row for row in before_after if row["delta_overall"] <= 0]
    return {
        "experiment_id": experiment["experiment_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "mock",
        "mock_notice": "This report compares deterministic mock base and mock fine-tuned responses on a tiny synthetic dataset.",
        "dataset": {"id": experiment["dataset_id"], "validation_examples": len(rows)},
        "training": experiment,
        "aggregate_metrics": {
            "base": base["aggregate"],
            "finetuned": finetuned["aggregate"],
            "delta_overall": round(finetuned["aggregate"]["overall"] - base["aggregate"]["overall"], 4),
        },
        "per_task_metrics": {"base": base["per_task"], "finetuned": finetuned["per_task"]},
        "before_after_examples": before_after,
        "failure_cases": failures,
        "qualitative_summary": "The mock fine-tuned model gives more domain-specific, grounded answers and explicitly avoids unsupported claims.",
        "limitations": [
            "Synthetic images and labels are placeholders for workflow validation.",
            "No real VLM weights are loaded or updated in mock mode.",
            "Expert-curated annotations and real visual grounding are required before claiming scientific performance.",
        ],
    }


def get_report(experiment_id: str) -> dict:
    path = OUTPUT_DIR / experiment_id / "comparison_report.json"
    if not path.exists():
        raise KeyError(f"No report for experiment: {experiment_id}")
    return json.loads(path.read_text(encoding="utf-8"))

