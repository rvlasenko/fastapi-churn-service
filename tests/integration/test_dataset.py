from fastapi.testclient import TestClient


def test_preview_default_returns_10_rows(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/preview")
    assert response.status_code == 200
    body = response.json()
    assert body["total_returned"] == 10
    assert len(body["rows"]) == 10


def test_preview_custom_n(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/preview?n=5")
    assert response.status_code == 200
    body = response.json()
    assert body["total_returned"] == 5
    assert len(body["rows"]) == 5


def test_preview_n_zero_fails(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/preview?n=0")
    assert response.status_code == 422


def test_preview_n_over_limit_fails(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/preview?n=501")
    assert response.status_code == 422


def test_preview_row_has_expected_fields(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/preview?n=1")
    assert response.status_code == 200
    row = response.json()["rows"][0]

    expected = {
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
        "region",
        "device_type",
        "payment_method",
        "autopay_enabled",
        "churn",
    }
    assert expected == set(row.keys())


def test_info_returns_200(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/info")
    assert response.status_code == 200


def test_info_has_required_fields(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/info")
    assert response.status_code == 200
    body = response.json()
    assert "row_count" in body
    assert "column_count" in body
    assert "feature_names" in body
    assert "churn_distribution" in body


def test_info_churn_not_in_feature_names(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/info")
    assert response.status_code == 200
    assert "churn" not in response.json()["feature_names"]


def test_info_distribution_sums_to_row_count(client: TestClient) -> None:
    response = client.get("/api/v1/dataset/info")
    assert response.status_code == 200
    body = response.json()
    dist = body["churn_distribution"]
    assert dist["retained"] + dist["churned"] == body["row_count"]
