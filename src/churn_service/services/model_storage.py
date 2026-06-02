from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from churn_service.core.exceptions import ModelLoadError, ModelNotFoundError

_MODEL_FILENAME = "churn_model.joblib"


@dataclass
class TrainedModel:
    pipeline: Pipeline
    trained_at: datetime
    model_type: str
    accuracy: float
    f1: float
    train_size: int
    test_size: int


class ModelStorageService:
    """Owns persistence of the trained churn model.

    Model is loaded from disk in __init__ so it is available immediately
    after the service is created (at application startup via lifespan).
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._path = models_dir / _MODEL_FILENAME
        self._current: TrainedModel | None = self._load_if_exists()

    @property
    def current(self) -> TrainedModel | None:
        return self._current

    def exists(self) -> bool:
        return self._path.exists()

    def save(self, trained_model: TrainedModel) -> None:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_model, self._path)
        self._current = trained_model

    def load(self) -> TrainedModel:
        if not self.exists():
            raise ModelNotFoundError(f"Model file not found: {self._path}")
        try:
            return joblib.load(self._path)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model from {self._path}") from exc

    def _load_if_exists(self) -> TrainedModel | None:
        if not self.exists():
            return None
        try:
            return joblib.load(self._path)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model from {self._path}") from exc
