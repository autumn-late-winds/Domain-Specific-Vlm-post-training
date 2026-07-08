from fastapi import APIRouter

from app.schemas.common import InferenceRequest
from app.services.mock_vlm import MockVLM

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/query")
def query(request: InferenceRequest):
    model = MockVLM("finetuned" if request.model_variant == "finetuned" else "base")
    return {
        "model_variant": request.model_variant,
        "task_type": request.task_type,
        "response": model.generate(request.question, request.task_type, {"modality": request.image_name or "uploaded image"}),
        "mock": True,
    }

