from fastapi import APIRouter, Response

from app.services.evaluation_service import get_report
from app.services.report_service import report_markdown

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{experiment_id}/json")
def json_report(experiment_id: str):
    return get_report(experiment_id)


@router.get("/{experiment_id}/markdown")
def markdown_report(experiment_id: str):
    return Response(report_markdown(experiment_id), media_type="text/markdown")

