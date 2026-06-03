from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn_service.schemas.training import ModelType, TrainingConfigChurn
from churn_service.services.pipeline import build_churn_pipeline
from churn_service.services.preprocessing import RANDOM_STATE


def _classifier(config: TrainingConfigChurn):
    pipeline = build_churn_pipeline(config)
    return pipeline.named_steps["classifier"]


def test_default_config_returns_logreg() -> None:
    clf = _classifier(TrainingConfigChurn())
    assert isinstance(clf, LogisticRegression)


def test_logreg_config_returns_logreg() -> None:
    clf = _classifier(TrainingConfigChurn(model_type=ModelType.LOGREG))
    assert isinstance(clf, LogisticRegression)


def test_random_forest_config_returns_rf() -> None:
    clf = _classifier(TrainingConfigChurn(model_type=ModelType.RANDOM_FOREST))
    assert isinstance(clf, RandomForestClassifier)


def test_default_random_state_applied_to_logreg() -> None:
    clf = _classifier(TrainingConfigChurn())
    assert clf.random_state == RANDOM_STATE


def test_default_random_state_applied_to_rf() -> None:
    clf = _classifier(TrainingConfigChurn(model_type=ModelType.RANDOM_FOREST))
    assert clf.random_state == RANDOM_STATE


def test_default_max_iter_applied_to_logreg() -> None:
    clf = _classifier(TrainingConfigChurn())
    assert clf.max_iter == 1000


def test_custom_max_iter_overrides_default() -> None:
    clf = _classifier(TrainingConfigChurn(hyperparameters={"max_iter": 500}))
    assert clf.max_iter == 500


def test_custom_n_estimators_applied_to_rf() -> None:
    clf = _classifier(
        TrainingConfigChurn(
            model_type=ModelType.RANDOM_FOREST, hyperparameters={"n_estimators": 50}
        )
    )
    assert clf.n_estimators == 50


def test_pipeline_has_preprocessor_and_classifier_steps() -> None:
    pipeline = build_churn_pipeline(TrainingConfigChurn())
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
