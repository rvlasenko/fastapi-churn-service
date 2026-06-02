import pytest
from fastapi.testclient import TestClient

from churn_service.core.config import Settings
from churn_service.dependencies import (
    get_dataset_service,
    get_model_storage_service,
    get_model_training_service,
    get_prediction_service,
    get_preprocessing_service,
)
from churn_service.main import create_app
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.prediction import PredictionService
from churn_service.services.preprocessing import PreprocessingService
from churn_service.services.training import ModelTrainingService

# ---------------------------------------------------------------------------
# Module-scoped fixtures: isolated fresh storage independent of other tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fresh_storage(tmp_path_factory) -> ModelStorageService:
    return ModelStorageService(tmp_path_factory.mktemp("models_status"))


@pytest.fixture(scope="module")
def fresh_training_service(
    preprocessing_service: PreprocessingService,
    fresh_storage: ModelStorageService,
) -> ModelTrainingService:
    return ModelTrainingService(preprocessing_service, fresh_storage)


@pytest.fixture(scope="module")
def status_client(
    test_settings: Settings,
    dataset_service,
    preprocessing_service: PreprocessingService,
    fresh_storage: ModelStorageService,
    fresh_training_service: ModelTrainingService,
    prediction_service: PredictionService,
) -> TestClient:
    application = create_app(settings=test_settings)
    application.dependency_overrides[get_dataset_service] = lambda: dataset_service
    application.dependency_overrides[get_preprocessing_service] = lambda: preprocessing_service
    application.dependency_overrides[get_model_storage_service] = lambda: fresh_storage
    application.dependency_overrides[get_model_training_service] = lambda: fresh_training_service
    application.dependency_overrides[get_prediction_service] = lambda: prediction_service
    with TestClient(application) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests — ordered: "not trained" checks must run before POST /train is called
# ---------------------------------------------------------------------------


def test_status_returns_200(status_client: TestClient) -> None:
    response = status_client.get("/api/v1/model/status")
    assert response.status_code == 200


def test_status_not_trained_initially(status_client: TestClient) -> None:
    response = status_client.get("/api/v1/model/status")
    body = response.json()
    assert body["is_trained"] is False
    assert body["trained_at"] is None
    assert body["metrics"] is None


def test_status_trained_after_train_call(status_client: TestClient) -> None:
    status_client.post("/api/v1/model/train")
    response = status_client.get("/api/v1/model/status")
    assert response.status_code == 200
    assert response.json()["is_trained"] is True


def test_status_metrics_match_train_response(status_client: TestClient) -> None:
    train_body = status_client.post("/api/v1/model/train").json()
    status_body = status_client.get("/api/v1/model/status").json()
    assert status_body["metrics"]["accuracy"] == train_body["accuracy"]
    assert status_body["metrics"]["f1"] == train_body["f1"]
    assert status_body["metrics"]["train_size"] == train_body["train_size"]
    assert status_body["metrics"]["test_size"] == train_body["test_size"]


# ---------------------------------------------------------------------------
# Startup loading test — verifies that lifespan loads an existing model file
# ---------------------------------------------------------------------------


def test_startup_loads_existing_model_without_retraining(
    tmp_path,
    test_settings: Settings,
    preprocessing_service: PreprocessingService,
) -> None:
    models_dir = tmp_path / "models"

    # Pre-populate model file by training directly (no HTTP call)
    storage = ModelStorageService(models_dir)
    training_service = ModelTrainingService(preprocessing_service, storage)
    training_service.train_and_save()
    assert (models_dir / "churn_model.joblib").exists()

    # Create fresh app pointing to that models dir — no dependency overrides
    settings_with_model = Settings(
        app_name="test-restart",
        debug=True,
        models_dir=models_dir,
    )
    fresh_app = create_app(settings=settings_with_model)

    with TestClient(fresh_app) as client:
        response = client.get("/api/v1/model/status")

    assert response.status_code == 200
    body = response.json()
    assert body["is_trained"] is True
    assert body["trained_at"] is not None
    assert body["metrics"] is not None
