import pytest
from fastapi.testclient import TestClient

from churn_service.core.config import Settings
from churn_service.dependencies import (
    get_dataset_service,
    get_model_storage_service,
    get_prediction_service,
    get_preprocessing_service,
)
from churn_service.main import create_app
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.prediction import PredictionService
from churn_service.services.preprocessing import PreprocessingService
from churn_service.services.training import ModelTrainingService
from tests.conftest import VALID_FEATURE_PAYLOAD

VALID_PREDICT_PAYLOAD = {"items": [VALID_FEATURE_PAYLOAD]}

HIGH_RISK_PAYLOAD = {
    "monthly_fee": 9.99,
    "usage_hours": 1.0,
    "support_requests": 20,
    "account_age_months": 1,
    "failed_payments": 10,
    "region": "africa",
    "device_type": "tablet",
    "payment_method": "crypto",
    "autopay_enabled": 0,
}

VALID_BATCH_PAYLOAD = {"items": [VALID_FEATURE_PAYLOAD, HIGH_RISK_PAYLOAD]}


def _make_predict_client(
    test_settings: Settings,
    dataset_service,
    preprocessing_service: PreprocessingService,
    storage: ModelStorageService,
) -> TestClient:
    prediction_svc = PredictionService(storage)
    application = create_app(settings=test_settings)
    application.dependency_overrides[get_dataset_service] = lambda: dataset_service
    application.dependency_overrides[get_preprocessing_service] = lambda: preprocessing_service
    application.dependency_overrides[get_model_storage_service] = lambda: storage
    application.dependency_overrides[get_prediction_service] = lambda: prediction_svc
    return TestClient(application)


@pytest.fixture(scope="module")
def untrained_predict_client(
    test_settings: Settings,
    dataset_service,
    preprocessing_service: PreprocessingService,
    tmp_path_factory,
) -> TestClient:
    storage = ModelStorageService(tmp_path_factory.mktemp("predict_no_model"))
    with _make_predict_client(test_settings, dataset_service, preprocessing_service, storage) as c:
        yield c


@pytest.fixture(scope="module")
def trained_predict_client(
    test_settings: Settings,
    dataset_service,
    preprocessing_service: PreprocessingService,
    tmp_path_factory,
) -> TestClient:
    storage = ModelStorageService(tmp_path_factory.mktemp("predict_models"))
    ModelTrainingService(preprocessing_service, storage).train_and_save()
    with _make_predict_client(test_settings, dataset_service, preprocessing_service, storage) as c:
        yield c


# ---------------------------------------------------------------------------
# 503 — no trained model
# ---------------------------------------------------------------------------


def test_predict_without_model_returns_503(untrained_predict_client: TestClient) -> None:
    response = untrained_predict_client.post("/api/v1/predict/", json=VALID_PREDICT_PAYLOAD)
    assert response.status_code == 503
    assert "train" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


def test_single_predict_returns_200(trained_predict_client: TestClient) -> None:
    response = trained_predict_client.post("/api/v1/predict/", json=VALID_PREDICT_PAYLOAD)
    assert response.status_code == 200


def test_single_predict_response_has_required_fields(trained_predict_client: TestClient) -> None:
    response = trained_predict_client.post("/api/v1/predict/", json=VALID_PREDICT_PAYLOAD)
    item = response.json()["predictions"][0]
    assert "predicted_class" in item
    assert "churn_probability" in item
    assert "retained_probability" in item


def test_single_predict_class_is_binary(trained_predict_client: TestClient) -> None:
    response = trained_predict_client.post("/api/v1/predict/", json=VALID_PREDICT_PAYLOAD)
    assert response.json()["predictions"][0]["predicted_class"] in (0, 1)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


def test_batch_predict_returns_one_result_per_input(trained_predict_client: TestClient) -> None:
    response = trained_predict_client.post("/api/v1/predict/", json=VALID_BATCH_PAYLOAD)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_batch_predict_preserves_order(trained_predict_client: TestClient) -> None:
    # Send the same two items reversed and confirm results flip accordingly.
    forward = trained_predict_client.post("/api/v1/predict/", json=VALID_BATCH_PAYLOAD).json()
    reversed_payload = {"items": [HIGH_RISK_PAYLOAD, VALID_FEATURE_PAYLOAD]}
    backward = trained_predict_client.post("/api/v1/predict/", json=reversed_payload).json()

    assert forward["predictions"][0] == backward["predictions"][1]
    assert forward["predictions"][1] == backward["predictions"][0]


# ---------------------------------------------------------------------------
# Probability invariants
# ---------------------------------------------------------------------------


def test_probabilities_sum_to_one(trained_predict_client: TestClient) -> None:
    response = trained_predict_client.post("/api/v1/predict/", json=VALID_BATCH_PAYLOAD)
    for item in response.json()["predictions"]:
        total = item["churn_probability"] + item["retained_probability"]
        assert abs(total - 1.0) < 1e-6


def test_probabilities_are_between_zero_and_one(trained_predict_client: TestClient) -> None:
    response = trained_predict_client.post("/api/v1/predict/", json=VALID_BATCH_PAYLOAD)
    for item in response.json()["predictions"]:
        assert 0.0 <= item["churn_probability"] <= 1.0
        assert 0.0 <= item["retained_probability"] <= 1.0


# ---------------------------------------------------------------------------
# Input validation (Pydantic boundary — applies regardless of model state)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("region", "narnia"),
        ("device_type", "smartwatch"),
        ("payment_method", "bitcoin"),
        ("monthly_fee", 0),
        ("monthly_fee", -5),
        ("usage_hours", -1),
        ("usage_hours", 800),
        ("autopay_enabled", 2),
    ],
)
def test_predict_rejects_invalid_field(
    trained_predict_client: TestClient, field: str, value
) -> None:
    payload = {"items": [{**VALID_FEATURE_PAYLOAD, field: value}]}
    response = trained_predict_client.post("/api/v1/predict/", json=payload)
    assert response.status_code == 422


def test_predict_rejects_empty_items(trained_predict_client: TestClient) -> None:
    response = trained_predict_client.post("/api/v1/predict/", json={"items": []})
    assert response.status_code == 422
