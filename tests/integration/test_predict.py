import pytest
from fastapi.testclient import TestClient

from tests.conftest import VALID_FEATURE_PAYLOAD


def test_predict_echoes_valid_input(client: TestClient) -> None:
    response = client.post("/api/v1/predict/", json=VALID_FEATURE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert data["input"]["monthly_fee"] == VALID_FEATURE_PAYLOAD["monthly_fee"]
    assert data["input"]["region"] == "europe"
    assert data["input"]["device_type"] == "mobile"


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
def test_predict_rejects_invalid_field(client: TestClient, field: str, value) -> None:
    payload = {**VALID_FEATURE_PAYLOAD, field: value}
    response = client.post("/api/v1/predict/", json=payload)
    assert response.status_code == 422
