from fastapi import APIRouter, Depends

from churn_service.dependencies import get_model_storage_service, get_model_training_service
from churn_service.schemas.model import ModelMetrics, ModelStatusResponse
from churn_service.schemas.training import TrainResponse
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.training import ModelTrainingService

router = APIRouter()


@router.post("/train")
def train(
    service: ModelTrainingService = Depends(get_model_training_service),  # noqa: B008
) -> TrainResponse:
    return service.train_and_save()


@router.get("/status")
def status(
    storage: ModelStorageService = Depends(get_model_storage_service),  # noqa: B008
) -> ModelStatusResponse:
    trained_model = storage.current
    if trained_model is None:
        return ModelStatusResponse(is_trained=False, trained_at=None, metrics=None)
    return ModelStatusResponse(
        is_trained=True,
        trained_at=trained_model.trained_at,
        metrics=ModelMetrics(
            accuracy=trained_model.accuracy,
            f1=trained_model.f1,
            train_size=trained_model.train_size,
            test_size=trained_model.test_size,
        ),
    )
