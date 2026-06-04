from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query

from churn_service.dependencies import (
    get_model_storage_service,
    get_model_training_service,
    get_training_history_service,
)
from churn_service.schemas.history import ModelMetricsResponse
from churn_service.schemas.model import ModelMetrics, ModelSchemaResponse, ModelStatusResponse
from churn_service.schemas.training import ModelType, TrainingConfigChurn, TrainResponse
from churn_service.services.model_schema import build_model_schema
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.training import ModelTrainingService
from churn_service.services.training_history import TrainingHistoryService

router = APIRouter()


@router.get("/schema")
def schema() -> ModelSchemaResponse:
    return build_model_schema()


@router.post("/train")
def train(
    config: TrainingConfigChurn | None = Body(default=None),  # noqa: B008
    service: ModelTrainingService = Depends(get_model_training_service),  # noqa: B008
) -> TrainResponse:
    return service.train_and_save(config or TrainingConfigChurn())


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
            roc_auc=trained_model.roc_auc,
            train_size=trained_model.train_size,
            test_size=trained_model.test_size,
        ),
        model_type=trained_model.model_type,
        hyperparameters=trained_model.hyperparameters,
    )


@router.get("/metrics")
def metrics(
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    model_type: ModelType | None = None,
    history: TrainingHistoryService = Depends(get_training_history_service),  # noqa: B008
) -> ModelMetricsResponse:
    records = history.load(
        model_type=model_type.value if model_type is not None else None,
        limit=limit,
    )
    return ModelMetricsResponse(
        latest=records[0] if records else None,
        history=records,
    )
