import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from churn_service.core.exceptions import ModelLoadError, ModelNotTrainedError

logger = logging.getLogger(__name__)

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
    roc_auc: float | None = None
    hyperparameters: dict[str, int | float | str | bool | None] = field(default_factory=dict)


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
        logger.info("Model saved: model_type=%s path=%s", trained_model.model_type, self._path)

    def load(self) -> TrainedModel:
        if not self.exists():
            raise ModelNotTrainedError(f"Model file not found: {self._path}")
        return self._load_and_validate()

    def _load_if_exists(self) -> TrainedModel | None:
        if not self.exists():
            logger.info("No model artifact found: path=%s", self._path)
            return None
        model = self._load_and_validate()
        logger.info("Model artifact loaded: model_type=%s", model.model_type)
        return model

    def _load_and_validate(self) -> TrainedModel:
        try:
            loaded = joblib.load(self._path)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model from {self._path}") from exc
        if not isinstance(loaded, TrainedModel):
            raise ModelLoadError(
                f"Expected TrainedModel, got {type(loaded).__name__} in {self._path}"
            )
        return loaded
