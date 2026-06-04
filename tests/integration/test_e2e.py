"""
End-to-end happy path: dataset → train → status → metrics → predict.

Uses module-scoped fixtures with isolated storage so this test is independent
of execution order and does not share state with other integration tests.
"""

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

_PREDICT_PAYLOAD = {
    "items": [
        {
            "monthly_fee": 29.99,
            "usage_hours": 120.5,
            "support_requests": 3,
            "account_age_months": 24,
            "failed_payments": 0,
            "region": "europe",
            "device_type": "mobile",
            "payment_method": "card",
            "autopay_enabled": 1,
        }
    ]
}


@pytest.fixture(scope="module")
def e2e_client(
    tmp_path_factory,
    test_settings: Settings,
    dataset_service,
    preprocessing_service: PreprocessingService,
) -> TestClient:
    storage = ModelStorageService(tmp_path_factory.mktemp("models_e2e"))
    history = TrainingHistoryService(tmp_path_factory.mktemp("history_e2e"))
    training = ModelTrainingService(preprocessing_service, storage, history)
    prediction = PredictionService(storage)

    application = create_app(settings=test_settings)
    application.dependency_overrides[get_dataset_service] = lambda: dataset_service
    application.dependency_overrides[get_preprocessing_service] = lambda: preprocessing_service
    application.dependency_overrides[get_model_storage_service] = lambda: storage
    application.dependency_overrides[get_model_training_service] = lambda: training
    application.dependency_overrides[get_training_history_service] = lambda: history
    application.dependency_overrides[get_prediction_service] = lambda: prediction

    with TestClient(application) as c:
        yield c


def test_e2e_happy_path(e2e_client: TestClient) -> None:
    # 1. Dataset info
    info_response = e2e_client.get("/api/v1/dataset/info")
    assert info_response.status_code == 200
    info = info_response.json()
    assert info["row_count"] > 0
    assert "churn" not in info["feature_names"]

    # 2. Train
    train_response = e2e_client.post("/api/v1/model/train")
    assert train_response.status_code == 200
    train = train_response.json()
    assert 0.0 <= train["accuracy"] <= 1.0
    assert train["train_size"] + train["test_size"] == info["row_count"]

    # 3. Status reflects the trained model
    status_response = e2e_client.get("/api/v1/model/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["is_trained"] is True
    assert status["metrics"]["accuracy"] == train["accuracy"]
    assert status["metrics"]["roc_auc"] == train["roc_auc"]

    # 4. Metrics history contains the training run
    metrics_response = e2e_client.get("/api/v1/model/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["latest"] is not None
    assert len(metrics["history"]) == 1
    assert metrics["history"][0]["accuracy"] == train["accuracy"]

    # 5. Predict returns a valid probability distribution
    predict_response = e2e_client.post("/api/v1/predict/", json=_PREDICT_PAYLOAD)
    assert predict_response.status_code == 200
    predictions = predict_response.json()["predictions"]
    assert len(predictions) == 1
    pred = predictions[0]
    assert pred["predicted_class"] in {0, 1}
    assert abs(pred["churn_probability"] + pred["retained_probability"] - 1.0) < 1e-6
