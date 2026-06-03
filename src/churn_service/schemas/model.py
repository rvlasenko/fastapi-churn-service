from datetime import datetime

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    accuracy: float
    f1: float
    train_size: int
    test_size: int


class ModelStatusResponse(BaseModel):
    is_trained: bool
    trained_at: datetime | None
    metrics: ModelMetrics | None
    model_type: str | None = None
    hyperparameters: dict[str, int | float | str | bool | None] | None = None
