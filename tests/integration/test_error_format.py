import pytest
from fastapi.testclient import TestClient

from churn_service.dependencies import (
    get_dataset_service,
    get_model_storage_service,
    get_prediction_service,
    get_preprocessing_service,
)
from churn_service.main import create_app
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.prediction import PredictionService
from tests.conftest import VALID_FEATURE_PAYLOAD


@pytest.fixture
def untrained_client(
    test_settings,
    dataset_service,
    preprocessing_service,
    tmp_path_factory,
) -> TestClient:
    storage = ModelStorageService(tmp_path_factory.mktemp("error_format_models"))
    prediction_svc = PredictionService(storage)
    application = create_app(settings=test_settings)
    application.dependency_overrides[get_dataset_service] = lambda: dataset_service
    application.dependency_overrides[get_preprocessing_service] = lambda: preprocessing_service
    application.dependency_overrides[get_model_storage_service] = lambda: storage
    application.dependency_overrides[get_prediction_service] = lambda: prediction_svc
    with TestClient(application) as c:
        yield c


# ---------------------------------------------------------------------------
# Shape invariant — every error response has {error: {code, message, details}}
# ---------------------------------------------------------------------------


def test_503_response_has_error_format(untrained_client: TestClient) -> None:
    response = untrained_client.post("/api/v1/predict/", json={"items": [VALID_FEATURE_PAYLOAD]})
    assert response.status_code == 503
    body = response.json()
    assert "error" in body
    assert "detail" not in body
    error = body["error"]
    assert "code" in error
    assert "message" in error
    assert "details" in error


def test_422_response_has_error_format(untrained_client: TestClient) -> None:
    payload = {"items": [{**VALID_FEATURE_PAYLOAD, "region": "narnia"}]}
    response = untrained_client.post("/api/v1/predict/", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert "detail" not in body
    error = body["error"]
    assert "code" in error
    assert "message" in error
    assert "details" in error


def test_404_response_has_error_format(untrained_client: TestClient) -> None:
    response = untrained_client.get("/api/v1/nonexistent-endpoint")
    assert response.status_code == 404
    body = response.json()
    assert "error" in body
    assert "detail" not in body


# ---------------------------------------------------------------------------
# Stable error codes
# ---------------------------------------------------------------------------


def test_503_model_not_trained_error_code(untrained_client: TestClient) -> None:
    response = untrained_client.post("/api/v1/predict/", json={"items": [VALID_FEATURE_PAYLOAD]})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_trained"


def test_422_validation_error_code(untrained_client: TestClient) -> None:
    payload = {"items": [{**VALID_FEATURE_PAYLOAD, "region": "narnia"}]}
    response = untrained_client.post("/api/v1/predict/", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_422_invalid_hyperparameter_error_code(untrained_client: TestClient) -> None:
    payload = {"model_type": "random_forest", "hyperparameters": {"n_estimators": -1}}
    response = untrained_client.post("/api/v1/model/train", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_404_not_found_error_code(untrained_client: TestClient) -> None:
    response = untrained_client.get("/api/v1/nonexistent-endpoint")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# Validation details are a list; 503/404 details are null
# ---------------------------------------------------------------------------


def test_422_details_is_list(untrained_client: TestClient) -> None:
    payload = {"items": [{**VALID_FEATURE_PAYLOAD, "region": "narnia"}]}
    response = untrained_client.post("/api/v1/predict/", json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["error"]["details"], list)
    assert len(response.json()["error"]["details"]) > 0


def test_503_details_is_null(untrained_client: TestClient) -> None:
    response = untrained_client.post("/api/v1/predict/", json={"items": [VALID_FEATURE_PAYLOAD]})
    assert response.status_code == 503
    assert response.json()["error"]["details"] is None


# ---------------------------------------------------------------------------
# No traceback or internal repr in responses
# ---------------------------------------------------------------------------


def test_error_response_has_no_traceback(untrained_client: TestClient) -> None:
    response = untrained_client.post("/api/v1/predict/", json={"items": [VALID_FEATURE_PAYLOAD]})
    text = response.text
    assert "Traceback" not in text
    assert 'File "' not in text
    assert "Exception" not in text
