from __future__ import annotations

import logging
from datetime import UTC, datetime

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from churn_service.schemas.history import TrainingRecord
from churn_service.schemas.training import TrainingConfigChurn, TrainResponse
from churn_service.services.model_storage import ModelStorageService, TrainedModel
from churn_service.services.pipeline import build_churn_pipeline
from churn_service.services.preprocessing import PreprocessingService
from churn_service.services.training_history import TrainingHistoryService

logger = logging.getLogger(__name__)


class ModelTrainingService:
    def __init__(
        self,
        preprocessing: PreprocessingService,
        storage: ModelStorageService,
        history: TrainingHistoryService,
    ) -> None:
        self._preprocessing = preprocessing
        self._storage = storage
        self._history = history

    def train(self, config: TrainingConfigChurn | None = None) -> TrainedModel:
        effective_config = config or TrainingConfigChurn()
        split = self._preprocessing.prepare_split()
        pipeline = build_churn_pipeline(effective_config)
        pipeline.fit(split.X_train, split.y_train)  # noqa: N806
        y_pred = pipeline.predict(split.X_test)  # noqa: N806
        y_proba = pipeline.predict_proba(split.X_test)[:, 1]  # noqa: N806

        try:
            roc_auc: float | None = round(float(roc_auc_score(split.y_test, y_proba)), 4)
        except ValueError:
            roc_auc = None  # single-class test split

        return TrainedModel(
            pipeline=pipeline,
            trained_at=datetime.now(tz=UTC),
            model_type=effective_config.model_type.value,
            hyperparameters=dict(effective_config.hyperparameters),
            accuracy=round(float(accuracy_score(split.y_test, y_pred)), 4),
            f1=round(float(f1_score(split.y_test, y_pred)), 4),
            roc_auc=roc_auc,
            train_size=len(split.y_train),
            test_size=len(split.y_test),
        )

    def train_and_save(self, config: TrainingConfigChurn | None = None) -> TrainResponse:
        effective_config = config or TrainingConfigChurn()
        logger.info("Training started: model_type=%s", effective_config.model_type.value)

        try:
            trained_model = self.train(config)
        except Exception:
            logger.exception("Training failed: model_type=%s", effective_config.model_type.value)
            raise

        try:
            self._storage.save(trained_model)
        except Exception:
            logger.exception("Model save failed: model_type=%s", effective_config.model_type.value)
            raise

        record = TrainingRecord(
            trained_at=trained_model.trained_at,
            model_type=trained_model.model_type,
            hyperparameters=trained_model.hyperparameters,
            accuracy=trained_model.accuracy,
            f1=trained_model.f1,
            roc_auc=trained_model.roc_auc,
            train_size=trained_model.train_size,
            test_size=trained_model.test_size,
        )
        try:
            self._history.append(record)
        except Exception:
            logger.exception("History append failed: model_type=%s", effective_config.model_type.value)
            raise

        logger.info(
            "Training succeeded: model_type=%s accuracy=%s f1=%s roc_auc=%s",
            trained_model.model_type,
            trained_model.accuracy,
            trained_model.f1,
            trained_model.roc_auc,
        )
        return TrainResponse(
            accuracy=trained_model.accuracy,
            f1=trained_model.f1,
            roc_auc=trained_model.roc_auc,
            train_size=trained_model.train_size,
            test_size=trained_model.test_size,
        )
