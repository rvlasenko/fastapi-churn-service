from __future__ import annotations

from datetime import UTC, datetime

from sklearn.metrics import accuracy_score, f1_score

from churn_service.schemas.training import TrainingConfigChurn, TrainResponse
from churn_service.services.model_storage import ModelStorageService, TrainedModel
from churn_service.services.pipeline import build_churn_pipeline
from churn_service.services.preprocessing import PreprocessingService


class ModelTrainingService:
    def __init__(self, preprocessing: PreprocessingService, storage: ModelStorageService) -> None:
        self._preprocessing = preprocessing
        self._storage = storage

    def train(self, config: TrainingConfigChurn | None = None) -> TrainedModel:
        effective_config = config or TrainingConfigChurn()
        split = self._preprocessing.prepare_split()
        pipeline = build_churn_pipeline(effective_config)
        pipeline.fit(split.X_train, split.y_train)  # noqa: N806
        y_pred = pipeline.predict(split.X_test)  # noqa: N806

        return TrainedModel(
            pipeline=pipeline,
            trained_at=datetime.now(tz=UTC),
            model_type=effective_config.model_type.value,
            hyperparameters=dict(effective_config.hyperparameters),
            accuracy=round(float(accuracy_score(split.y_test, y_pred)), 4),
            f1=round(float(f1_score(split.y_test, y_pred)), 4),
            train_size=len(split.y_train),
            test_size=len(split.y_test),
        )

    def train_and_save(self, config: TrainingConfigChurn | None = None) -> TrainResponse:
        trained_model = self.train(config)
        self._storage.save(trained_model)
        return TrainResponse(
            accuracy=trained_model.accuracy,
            f1=trained_model.f1,
            train_size=trained_model.train_size,
            test_size=trained_model.test_size,
        )
