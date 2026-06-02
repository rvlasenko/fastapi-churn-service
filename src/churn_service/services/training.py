from datetime import UTC, datetime

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_service.schemas.training import TrainResponse
from churn_service.services.model_storage import ModelStorageService, TrainedModel
from churn_service.services.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    PreprocessingService,
)


class ModelTrainingService:
    def __init__(self, preprocessing: PreprocessingService, storage: ModelStorageService) -> None:
        self._preprocessing = preprocessing
        self._storage = storage

    def train(self) -> TrainedModel:
        split = self._preprocessing.prepare_split()
        pipeline = self._build_pipeline()
        pipeline.fit(split.X_train, split.y_train)  # noqa: N806
        y_pred = pipeline.predict(split.X_test)  # noqa: N806

        return TrainedModel(
            pipeline=pipeline,
            trained_at=datetime.now(tz=UTC),
            model_type=type(pipeline.named_steps["classifier"]).__name__,
            accuracy=round(float(accuracy_score(split.y_test, y_pred)), 4),
            f1=round(float(f1_score(split.y_test, y_pred)), 4),
            train_size=len(split.y_train),
            test_size=len(split.y_test),
        )

    def train_and_save(self) -> TrainResponse:
        trained_model = self.train()
        self._storage.save(trained_model)
        return TrainResponse(
            accuracy=trained_model.accuracy,
            f1=trained_model.f1,
            train_size=trained_model.train_size,
            test_size=trained_model.test_size,
        )

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            [
                ("numerical", StandardScaler(), NUMERICAL_FEATURES),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
            ]
        )
