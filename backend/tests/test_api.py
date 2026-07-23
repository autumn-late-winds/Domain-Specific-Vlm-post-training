import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "mock"


def test_dataset_examples():
    response = client.get("/datasets/demo_materials_vlm/examples")
    assert response.status_code == 200
    assert len(response.json()) > 0


def test_training_and_evaluation_api():
    train_response = client.post("/training/start", json={"dataset_id": "demo_materials_vlm", "mode": "mock", "epochs": 1})
    assert train_response.status_code == 200
    experiment_id = train_response.json()["experiment_id"]
    eval_response = client.post("/evaluation/run", json={"experiment_id": experiment_id, "dataset_id": "demo_materials_vlm"})
    assert eval_response.status_code == 200
    assert eval_response.json()["experiment_id"] == experiment_id

