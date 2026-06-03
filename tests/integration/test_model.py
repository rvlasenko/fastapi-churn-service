import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Existing tests — default behavior (no request body)
# ---------------------------------------------------------------------------


def test_train_returns_200(client: TestClient) -> None:
    response = client.post("/api/v1/model/train")
    assert response.status_code == 200


def test_train_response_has_required_fields(client: TestClient) -> None:
    response = client.post("/api/v1/model/train")
    assert response.status_code == 200
    body = response.json()
    assert "accuracy" in body
    assert "f1" in body
    assert "train_size" in body
    assert "test_size" in body


def test_accuracy_is_in_valid_range(client: TestClient) -> None:
    response = client.post("/api/v1/model/train")
    assert response.status_code == 200
    assert 0.0 <= response.json()["accuracy"] <= 1.0


def test_f1_is_in_valid_range(client: TestClient) -> None:
    response = client.post("/api/v1/model/train")
    assert response.status_code == 200
    assert 0.0 <= response.json()["f1"] <= 1.0


def test_train_sizes_match_split_info(client: TestClient) -> None:
    train_response = client.post("/api/v1/model/train")
    assert train_response.status_code == 200

    split_response = client.get("/api/v1/dataset/split-info")
    assert split_response.status_code == 200

    train_body = train_response.json()
    split_body = split_response.json()
    assert train_body["train_size"] == split_body["train_size"]
    assert train_body["test_size"] == split_body["test_size"]


# ---------------------------------------------------------------------------
# Model type selection
# ---------------------------------------------------------------------------


def test_train_with_logreg_config_returns_200(client: TestClient) -> None:
    response = client.post("/api/v1/model/train", json={"model_type": "logreg"})
    assert response.status_code == 200


def test_train_with_random_forest_config_returns_200(client: TestClient) -> None:
    response = client.post("/api/v1/model/train", json={"model_type": "random_forest"})
    assert response.status_code == 200


def test_train_with_invalid_model_type_returns_422(client: TestClient) -> None:
    response = client.post("/api/v1/model/train", json={"model_type": "xgboost"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Hyperparameter validation
# ---------------------------------------------------------------------------


def test_train_with_valid_logreg_hyperparameter_returns_200(client: TestClient) -> None:
    response = client.post(
        "/api/v1/model/train",
        json={"model_type": "logreg", "hyperparameters": {"max_iter": 500}},
    )
    assert response.status_code == 200


def test_train_with_valid_rf_hyperparameter_returns_200(client: TestClient) -> None:
    response = client.post(
        "/api/v1/model/train",
        json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 50}},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"model_type": "logreg", "hyperparameters": {"learning_rate": 0.1}},
        {"model_type": "random_forest", "hyperparameters": {"C": 1.0}},
        {"model_type": "logreg", "hyperparameters": {"learning_rate": 0.1, "C": 1.0}},
    ],
)
def test_train_with_unsupported_hyperparameter_returns_422(
    client: TestClient, payload: dict
) -> None:
    response = client.post("/api/v1/model/train", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        # logreg — C
        {"model_type": "logreg", "hyperparameters": {"C": "bad"}},
        {"model_type": "logreg", "hyperparameters": {"C": -1.0}},
        {"model_type": "logreg", "hyperparameters": {"C": 0}},
        # logreg — max_iter
        {"model_type": "logreg", "hyperparameters": {"max_iter": -100}},
        {"model_type": "logreg", "hyperparameters": {"max_iter": 1.5}},
        # logreg — solver
        {"model_type": "logreg", "hyperparameters": {"solver": "invalid_solver"}},
        # logreg — class_weight
        {"model_type": "logreg", "hyperparameters": {"class_weight": "wrong"}},
        # random_forest — n_estimators
        {"model_type": "random_forest", "hyperparameters": {"n_estimators": 0}},
        {"model_type": "random_forest", "hyperparameters": {"n_estimators": -10}},
        # random_forest — max_depth
        {"model_type": "random_forest", "hyperparameters": {"max_depth": -1}},
        # random_forest — min_samples_split
        {"model_type": "random_forest", "hyperparameters": {"min_samples_split": 1}},
        {"model_type": "random_forest", "hyperparameters": {"min_samples_split": 1.5}},
        # random_forest — min_samples_leaf
        {"model_type": "random_forest", "hyperparameters": {"min_samples_leaf": 0}},
        {"model_type": "random_forest", "hyperparameters": {"min_samples_leaf": 1.0}},
        # random_forest — class_weight
        {"model_type": "random_forest", "hyperparameters": {"class_weight": "invalid"}},
    ],
)
def test_train_with_invalid_hyperparameter_value_returns_422(
    client: TestClient, payload: dict
) -> None:
    response = client.post("/api/v1/model/train", json=payload)
    assert response.status_code == 422
