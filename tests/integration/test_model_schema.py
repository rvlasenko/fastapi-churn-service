import pytest
from fastapi.testclient import TestClient

from churn_service.dependencies import (
    get_dataset_service,
    get_model_storage_service,
    get_prediction_service,
    get_preprocessing_service,
)
from churn_service.main import create_app
from churn_service.schemas.features import (
    DeviceType,
    FeatureVectorChurn,
    PaymentMethod,
    Region,
)
from churn_service.services.model_schema import _resolve_field_type
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.prediction import PredictionService
from churn_service.services.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from churn_service.services.training import ModelTrainingService
from tests.conftest import VALID_FEATURE_PAYLOAD

SCHEMA_URL = "/api/v1/model/schema"


@pytest.fixture(scope="module")
def trained_schema_client(
    test_settings,
    dataset_service,
    preprocessing_service,
    tmp_path_factory,
) -> TestClient:
    storage = ModelStorageService(tmp_path_factory.mktemp("schema_models"))
    ModelTrainingService(preprocessing_service, storage).train_and_save()
    prediction_svc = PredictionService(storage)
    application = create_app(settings=test_settings)
    application.dependency_overrides[get_dataset_service] = lambda: dataset_service
    application.dependency_overrides[get_preprocessing_service] = lambda: preprocessing_service
    application.dependency_overrides[get_model_storage_service] = lambda: storage
    application.dependency_overrides[get_prediction_service] = lambda: prediction_svc
    with TestClient(application) as c:
        yield c


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_schema_returns_200(client: TestClient) -> None:
    response = client.get(SCHEMA_URL)
    assert response.status_code == 200


def test_schema_target(client: TestClient) -> None:
    response = client.get(SCHEMA_URL)
    assert response.json()["target"] == "churn"


def test_schema_no_ohe_columns(client: TestClient) -> None:
    body = client.get(SCHEMA_URL).text
    for prefix in ("region_", "device_type_", "payment_method_"):
        assert prefix not in body


# ---------------------------------------------------------------------------
# Internal consistency (constants)
# ---------------------------------------------------------------------------


def test_schema_numerical_feature_names(client: TestClient) -> None:
    response = client.get(SCHEMA_URL)
    names = [f["name"] for f in response.json()["numerical_features"]]
    assert names == NUMERICAL_FEATURES


def test_schema_categorical_feature_names(client: TestClient) -> None:
    response = client.get(SCHEMA_URL)
    names = [f["name"] for f in response.json()["categorical_features"]]
    assert names == CATEGORICAL_FEATURES


def test_schema_categorical_feature_type(client: TestClient) -> None:
    response = client.get(SCHEMA_URL)
    for feature in response.json()["categorical_features"]:
        assert feature["type"] == "str"


# ---------------------------------------------------------------------------
# Contract coherence (schema vs. FeatureVectorChurn)
# ---------------------------------------------------------------------------


def test_schema_fields_match_feature_vector_model(client: TestClient) -> None:
    data = client.get(SCHEMA_URL).json()
    schema_names = {f["name"] for f in data["numerical_features"]} | {
        f["name"] for f in data["categorical_features"]
    }
    assert schema_names == set(FeatureVectorChurn.model_fields.keys())


def test_schema_numerical_types_match_feature_vector_annotations(client: TestClient) -> None:
    fields = FeatureVectorChurn.model_fields
    for feature in client.get(SCHEMA_URL).json()["numerical_features"]:
        expected = _resolve_field_type(fields[feature["name"]].annotation)
        assert feature["type"] == expected


def test_schema_numerical_types_ground_truth(client: TestClient) -> None:
    types = {f["name"]: f["type"] for f in client.get(SCHEMA_URL).json()["numerical_features"]}
    assert types["monthly_fee"] == "float"
    assert types["usage_hours"] == "float"
    assert types["support_requests"] == "int"
    assert types["account_age_months"] == "int"
    assert types["failed_payments"] == "int"
    assert types["autopay_enabled"] == "int"


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


def test_schema_region_enum_values(client: TestClient) -> None:
    categorical = {
        f["name"]: f["accepted_values"]
        for f in client.get(SCHEMA_URL).json()["categorical_features"]
    }
    assert set(categorical["region"]) == {v.value for v in Region}


def test_schema_device_type_enum_values(client: TestClient) -> None:
    categorical = {
        f["name"]: f["accepted_values"]
        for f in client.get(SCHEMA_URL).json()["categorical_features"]
    }
    assert set(categorical["device_type"]) == {v.value for v in DeviceType}


def test_schema_payment_method_enum_values(client: TestClient) -> None:
    categorical = {
        f["name"]: f["accepted_values"]
        for f in client.get(SCHEMA_URL).json()["categorical_features"]
    }
    assert set(categorical["payment_method"]) == {v.value for v in PaymentMethod}


# ---------------------------------------------------------------------------
# End-to-end contract
# ---------------------------------------------------------------------------


def test_schema_enum_values_accepted_by_predict(trained_schema_client: TestClient) -> None:
    categorical = trained_schema_client.get(SCHEMA_URL).json()["categorical_features"]
    for feature in categorical:
        for value in feature["accepted_values"]:
            payload = {"items": [{**VALID_FEATURE_PAYLOAD, feature["name"]: value}]}
            response = trained_schema_client.post("/api/v1/predict/", json=payload)
            assert response.status_code == 200, (
                f"Predict rejected schema value {feature['name']}={value!r}: {response.json()}"
            )


def test_prediction_uses_same_feature_order(trained_schema_client: TestClient) -> None:
    response = trained_schema_client.post(
        "/api/v1/predict/", json={"items": [VALID_FEATURE_PAYLOAD]}
    )
    assert response.status_code == 200
