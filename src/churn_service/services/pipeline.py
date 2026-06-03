from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_service.schemas.training import ModelType, TrainingConfigChurn
from churn_service.services.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
)

_LOGREG_DEFAULTS: dict[str, object] = {
    "random_state": RANDOM_STATE,
    "max_iter": 1000,
}
_RF_DEFAULTS: dict[str, object] = {
    "random_state": RANDOM_STATE,
}


def build_churn_pipeline(config: TrainingConfigChurn) -> Pipeline:
    """Return a fresh unfitted sklearn Pipeline for the given training config."""
    preprocessor = ColumnTransformer(
        [
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", _build_classifier(config)),
        ]
    )


def _build_classifier(config: TrainingConfigChurn) -> LogisticRegression | RandomForestClassifier:
    params = config.hyperparameters  # names already validated by TrainingConfigChurn
    if config.model_type == ModelType.LOGREG:
        return LogisticRegression(**{**_LOGREG_DEFAULTS, **params})
    return RandomForestClassifier(**{**_RF_DEFAULTS, **params})
