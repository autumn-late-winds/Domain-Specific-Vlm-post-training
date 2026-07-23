from fastapi import APIRouter, HTTPException

from app.services import dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets():
    return dataset_service.list_datasets()


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        return dataset_service.get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{dataset_id}/stats")
def stats(dataset_id: str):
    return dataset_service.dataset_stats(dataset_id)


@router.post("/{dataset_id}/validate")
def validate(dataset_id: str):
    return dataset_service.validate_dataset(dataset_id)


@router.get("/{dataset_id}/examples")
def examples(dataset_id: str, split: str = "val"):
    return dataset_service.get_examples(dataset_id, split)

