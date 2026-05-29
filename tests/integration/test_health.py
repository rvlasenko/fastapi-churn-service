from fastapi.testclient import TestClient


def test_root_returns_running_message(client: TestClient) -> None:
    response = client.get("/api/v1/")
    assert response.status_code == 200
    assert response.json() == {"message": "ml churn service is running"}
