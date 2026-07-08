from fastapi import APIRouter

from app.schemas.common import EvaluationRequest
from app.services.evaluation_service import evaluate_experiment, get_report

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run")
def run(request: EvaluationRequest):
    return evaluate_experiment(request.experiment_id, request.dataset_id)


@router.get("/reports")
def reports():
    return {"message": "Use /training/experiments to discover experiment IDs, then /evaluation/reports/{experiment_id}."}


@router.get("/reports/{experiment_id}")
def report(experiment_id: str):
    return get_report(experiment_id)


@router.get("/reports/{experiment_id}/examples")
def report_examples(experiment_id: str):
    return get_report(experiment_id)["before_after_examples"]

