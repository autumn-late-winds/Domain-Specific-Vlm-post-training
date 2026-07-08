import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import OUTPUT_DIR
from app.services.dataset_service import get_examples


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_mock_training(dataset_id: str, base_model: str, epochs: int = 3) -> dict:
    experiment_id = f"mock-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    out_dir = OUTPUT_DIR / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = get_examples(dataset_id, "train")
    val_rows = get_examples(dataset_id, "val")
    start_time = _now()
    logs = []
    for step in range(1, epochs + 1):
        loss = round(1.2 / step + 0.05, 4)
        val_score = round(0.42 + step * 0.12, 4)
        logs.append({"step": step, "loss": loss, "val_mock_score": val_score, "time": _now()})
        time.sleep(0.02)
    summary = {
        "experiment_id": experiment_id,
        "base_model": base_model,
        "training_method": "mock_sft_lora_simulation",
        "dataset_id": dataset_id,
        "dataset_size": len(train_rows),
        "validation_size": len(val_rows),
        "start_time": start_time,
        "end_time": _now(),
        "config": {"mode": "mock", "epochs": epochs, "adapter": "simulated_lora"},
        "output_dir": str(out_dir),
        "status": "completed",
        "mock_notice": "No model weights were trained. Metrics are deterministic and intended for portfolio/demo smoke tests.",
    }
    (out_dir / "training_logs.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    (out_dir / "adapter_metadata.json").write_text(
        json.dumps({"adapter_type": "mock-lora", "base_model": base_model}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def list_experiments() -> list[dict]:
    experiments = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        summary_path = path / "experiment_summary.json"
        if summary_path.exists():
            experiments.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return experiments


def get_experiment(experiment_id: str) -> dict:
    path = OUTPUT_DIR / experiment_id / "experiment_summary.json"
    if not path.exists():
        raise KeyError(f"Unknown experiment: {experiment_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_logs(experiment_id: str) -> list[dict]:
    path = OUTPUT_DIR / experiment_id / "training_logs.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

