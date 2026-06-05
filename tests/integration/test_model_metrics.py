import pytest
from fastapi.testclient import TestClient

from churn_service.core.config import Settings
from churn_service.dependencies import (
    get_dataset_service,
    get_model_storage_service,
    get_model_training_service,
    get_prediction_service,
    get_preprocessing_service,
    get_training_history_service,
)
from churn_service.main import create_app
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.prediction import PredictionService
from churn_service.services.preprocessing import PreprocessingService
from churn_service.services.training import ModelTrainingService
from churn_service.services.training_history import TrainingHistoryService

# ---------------------------------------------------------------------------
# Module-scoped isolated fixtures for ordered tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fresh_storage(tmp_path_factory) -> ModelStorageService:
    return ModelStorageService(tmp_path_factory.mktemp("models_metrics"))


@pytest.fixture(scope="module")
def fresh_history_service(tmp_path_factory) -> TrainingHistoryService:
    return TrainingHistoryService(tmp_path_factory.mktemp("history_metrics"))


@pytest.fixture(scope="module")
def fresh_training_service(
    preprocessing_service: PreprocessingService,
    fresh_storage: ModelStorageService,
    fresh_history_service: TrainingHistoryService,
) -> ModelTrainingService:
    return ModelTrainingService(preprocessing_service, fresh_storage, fresh_history_service)


@pytest.fixture(scope="module")
def metrics_client(
    test_settings: Settings,
    dataset_service,
    preprocessing_service: PreprocessingService,
    fresh_storage: ModelStorageService,
    fresh_training_service: ModelTrainingService,
    fresh_history_service: TrainingHistoryService,
    prediction_service: PredictionService,
) -> TestClient:
    application = create_app(settings=test_settings)
    application.dependency_overrides[get_dataset_service] = lambda: dataset_service
    application.dependency_overrides[get_preprocessing_service] = lambda: preprocessing_service
    application.dependency_overrides[get_model_storage_service] = lambda: fresh_storage
    application.dependency_overrides[get_model_training_service] = lambda: fresh_training_service
    application.dependency_overrides[get_training_history_service] = lambda: fresh_history_service
    application.dependency_overrides[get_prediction_service] = lambda: prediction_service
    with TestClient(application) as c:
        yield c


# ---------------------------------------------------------------------------
# Ordered tests — build history state incrementally
# ---------------------------------------------------------------------------


def test_metrics_empty_initially(metrics_client: TestClient) -> None:
    response = metrics_client.get("/api/v1/model/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["latest"] is None
    assert body["history"] == []


def test_metrics_after_first_train(metrics_client: TestClient) -> None:
    metrics_client.post("/api/v1/model/train", json={"model_type": "logreg"})
    response = metrics_client.get("/api/v1/model/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["latest"] is not None
    assert len(body["history"]) == 1
    assert body["latest"]["model_type"] == "logreg"


def test_metrics_roc_auc_in_record(metrics_client: TestClient) -> None:
    body = metrics_client.get("/api/v1/model/metrics").json()
    record = body["latest"]
    assert "roc_auc" in record
    assert record["roc_auc"] is not None
    assert 0.0 <= record["roc_auc"] <= 1.0


def test_metrics_after_second_train(metrics_client: TestClient) -> None:
    metrics_client.post("/api/v1/model/train", json={"model_type": "logreg"})
    body = metrics_client.get("/api/v1/model/metrics").json()
    assert len(body["history"]) == 2


def test_metrics_latest_matches_most_recent(metrics_client: TestClient) -> None:
    metrics_client.post("/api/v1/model/train", json={"model_type": "random_forest"})
    body = metrics_client.get("/api/v1/model/metrics").json()
    assert body["latest"]["model_type"] == "random_forest"
    assert body["history"][0]["model_type"] == "random_forest"


def test_metrics_limit_param(metrics_client: TestClient) -> None:
    # At this point there are at least 3 training records
    response = metrics_client.get("/api/v1/model/metrics?limit=1")
    assert response.status_code == 200
    body = response.json()
    assert len(body["history"]) == 1


def test_metrics_model_type_filter_logreg(metrics_client: TestClient) -> None:
    response = metrics_client.get("/api/v1/model/metrics?model_type=logreg&limit=100")
    assert response.status_code == 200
    body = response.json()
    assert all(r["model_type"] == "logreg" for r in body["history"])
    assert body["latest"]["model_type"] == "logreg"


def test_metrics_model_type_filter_random_forest(metrics_client: TestClient) -> None:
    response = metrics_client.get("/api/v1/model/metrics?model_type=random_forest&limit=100")
    assert response.status_code == 200
    body = response.json()
    assert all(r["model_type"] == "random_forest" for r in body["history"])


# ---------------------------------------------------------------------------
# Validation error cases — independent of shared state
# ---------------------------------------------------------------------------


def test_metrics_invalid_model_type_returns_422(metrics_client: TestClient) -> None:
    response = metrics_client.get("/api/v1/model/metrics?model_type=xgboost")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_metrics_limit_zero_returns_422(metrics_client: TestClient) -> None:
    response = metrics_client.get("/api/v1/model/metrics?limit=0")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_metrics_limit_too_large_returns_422(metrics_client: TestClient) -> None:
    response = metrics_client.get("/api/v1/model/metrics?limit=101")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# ---------------------------------------------------------------------------
# Corrupted history file — independent app instance
# ---------------------------------------------------------------------------


def test_metrics_corrupted_history_returns_503(
    tmp_path,
    test_settings: Settings,
    dataset_service,
    preprocessing_service: PreprocessingService,
    prediction_service: PredictionService,
) -> None:
    models_dir = tmp_path / "models_corrupt"
    models_dir.mkdir()
    history_file = models_dir / "training_history.json"
    history_file.write_text("this is not json at all", encoding="utf-8")

    corrupted_history = TrainingHistoryService(models_dir)
    fresh_storage = ModelStorageService(models_dir)
    fresh_training = ModelTrainingService(preprocessing_service, fresh_storage, corrupted_history)

    app = create_app(settings=test_settings)
    app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    app.dependency_overrides[get_preprocessing_service] = lambda: preprocessing_service
    app.dependency_overrides[get_model_storage_service] = lambda: fresh_storage
    app.dependency_overrides[get_model_training_service] = lambda: fresh_training
    app.dependency_overrides[get_training_history_service] = lambda: corrupted_history
    app.dependency_overrides[get_prediction_service] = lambda: prediction_service

    with TestClient(app) as c:
        response = c.get("/api/v1/model/metrics")

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "history_load_failed"
    assert "message" in body["error"]


def test_metrics_history_write_error_during_train_returns_503(
    tmp_path,
    test_settings: Settings,
    dataset_service,
    preprocessing_service: PreprocessingService,
    prediction_service: PredictionService,
) -> None:
    from unittest.mock import MagicMock

    from churn_service.core.exceptions import HistoryWriteError

    models_dir = tmp_path / "models_write_fail"
    models_dir.mkdir()

    broken_history = MagicMock(spec=TrainingHistoryService)
    broken_history.load.return_value = []
    broken_history.append.side_effect = HistoryWriteError("disk full")

    fresh_storage = ModelStorageService(models_dir)
    fresh_training = ModelTrainingService(preprocessing_service, fresh_storage, broken_history)

    app = create_app(settings=test_settings)
    app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    app.dependency_overrides[get_preprocessing_service] = lambda: preprocessing_service
    app.dependency_overrides[get_model_storage_service] = lambda: fresh_storage
    app.dependency_overrides[get_model_training_service] = lambda: fresh_training
    app.dependency_overrides[get_training_history_service] = lambda: broken_history
    app.dependency_overrides[get_prediction_service] = lambda: prediction_service

    with TestClient(app) as c:
        response = c.post("/api/v1/model/train")

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "history_write_failed"
    assert "message" in body["error"]
