from typing import Any

from pydantic import BaseModel, Field


class DatasetExample(BaseModel):
    id: str
    image: str
    task_type: str
    question: str
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingRequest(BaseModel):
    dataset_id: str = "demo_materials_vlm"
    mode: str = "mock"
    base_model: str = "mock-vlm-materials"
    epochs: int = 3


class EvaluationRequest(BaseModel):
    experiment_id: str
    dataset_id: str = "demo_materials_vlm"


class InferenceRequest(BaseModel):
    question: str
    task_type: str = "materials_qa"
    model_variant: str = "finetuned"
    image_name: str | None = None

