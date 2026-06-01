from fastapi.testclient import TestClient


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
