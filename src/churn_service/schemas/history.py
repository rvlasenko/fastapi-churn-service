from datetime import datetime

from pydantic import BaseModel


class TrainingRecord(BaseModel):
    trained_at: datetime
    model_type: str
    hyperparameters: dict[str, int | float | str | bool | None]
    accuracy: float
    f1: float
    roc_auc: float | None
    train_size: int
    test_size: int


class ModelMetricsResponse(BaseModel):
    latest: TrainingRecord | None
    history: list[TrainingRecord]
