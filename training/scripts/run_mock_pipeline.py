import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.evaluation_service import evaluate_experiment
from app.services.training_service import start_mock_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock Materials VLM training and evaluation pipeline.")
    parser.add_argument("--dataset_id", default="demo_materials_vlm")
    parser.add_argument("--base_model", default="mock-vlm-materials")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    summary = start_mock_training(args.dataset_id, args.base_model, args.epochs)
    report = evaluate_experiment(summary["experiment_id"], args.dataset_id)
    print(json.dumps({"experiment": summary, "report": report}, indent=2))


if __name__ == "__main__":
    main()
