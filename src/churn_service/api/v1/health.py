from fastapi import APIRouter, Depends
from starlette.requests import Request

from churn_service.dependencies import get_model_storage_service
from churn_service.schemas.health import HealthResponse
from churn_service.services.model_storage import ModelStorageService

router = APIRouter()


@router.get("/")
def health_check(
    request: Request,
    storage: ModelStorageService = Depends(get_model_storage_service),
) -> HealthResponse:
    dataset_loaded: bool = request.app.state.dataset_loaded
    model_loaded = storage.current is not None
    status = "ok" if (dataset_loaded and model_loaded) else "degraded"
    return HealthResponse(
        status=status,
        dataset_loaded=dataset_loaded,
        model_loaded=model_loaded,
    )
