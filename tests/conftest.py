import pytest
from fastapi.testclient import TestClient

from churn_service.core.config import Settings
from churn_service.dependencies import get_dataset_service
from churn_service.main import create_app
from churn_service.services.dataset import DatasetService


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(app_name="churn-service-test", debug=True)


@pytest.fixture(scope="session")
def dataset_service(test_settings: Settings) -> DatasetService:
    return DatasetService(test_settings.dataset_path)


@pytest.fixture(scope="session")
def app(test_settings: Settings, dataset_service: DatasetService):
    application = create_app(settings=test_settings)
    application.dependency_overrides[get_dataset_service] = lambda: dataset_service
    return application


@pytest.fixture(scope="session")
def client(app) -> TestClient:
    with TestClient(app) as c:
        yield c


VALID_FEATURE_PAYLOAD = {
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
