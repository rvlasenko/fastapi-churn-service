import pytest
from fastapi.testclient import TestClient

from churn_service.core.config import Settings
from churn_service.main import create_app


@pytest.fixture()
def isolated_client(tmp_path):
    """Fresh app with an empty models directory — no model artifact present."""
    settings = Settings(models_dir=tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


def test_health_returns_typed_response(client: TestClient) -> None:
    response = client.get("/api/v1/")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"status", "dataset_loaded", "model_loaded"}


def test_health_status_is_ok_or_degraded(client: TestClient) -> None:
    body = client.get("/api/v1/").json()
    assert body["status"] in {"ok", "degraded"}


def test_health_dataset_loaded_true_in_normal_app(client: TestClient) -> None:
    body = client.get("/api/v1/").json()
    assert body["dataset_loaded"] is True


def test_health_model_loaded_false_before_training(isolated_client: TestClient) -> None:
    body = isolated_client.get("/api/v1/").json()
    assert body["model_loaded"] is False


def test_health_status_degraded_without_model(isolated_client: TestClient) -> None:
    body = isolated_client.get("/api/v1/").json()
    assert body["status"] == "degraded"


def test_health_model_loaded_true_after_training(client: TestClient) -> None:
    client.post("/api/v1/model/train", json={})
    body = client.get("/api/v1/").json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"
