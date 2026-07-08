from fastapi import APIRouter, HTTPException

from app.schemas.common import TrainingRequest
from app.services.training_service import get_experiment, get_logs, list_experiments, start_mock_training

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/start")
def start(request: TrainingRequest):
    if request.mode != "mock":
        raise HTTPException(status_code=400, detail="First-round MVP supports mock mode only.")
    return start_mock_training(request.dataset_id, request.base_model, request.epochs)


@router.get("/experiments")
def experiments():
    return list_experiments()


@router.get("/experiments/{experiment_id}")
def experiment(experiment_id: str):
    return get_experiment(experiment_id)


@router.get("/experiments/{experiment_id}/logs")
def logs(experiment_id: str):
    return get_logs(experiment_id)

