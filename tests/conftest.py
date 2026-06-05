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
from churn_service.services.dataset import DatasetService
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.prediction import PredictionService
from churn_service.services.preprocessing import PreprocessingService
from churn_service.services.training import ModelTrainingService
from churn_service.services.training_history import TrainingHistoryService


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory) -> Settings:
    return Settings(
        app_name="churn-service-test",
        debug=True,
        models_dir=tmp_path_factory.mktemp("models_settings"),
    )


@pytest.fixture(scope="session")
def dataset_service(test_settings: Settings) -> DatasetService:
    return DatasetService(test_settings.dataset_path)


@pytest.fixture(scope="session")
def preprocessing_service(dataset_service: DatasetService) -> PreprocessingService:
    return PreprocessingService(dataset_service.get_dataframe())


@pytest.fixture(scope="session")
def model_storage_service(tmp_path_factory) -> ModelStorageService:
    models_dir = tmp_path_factory.mktemp("models_session")
    return ModelStorageService(models_dir)


@pytest.fixture(scope="session")
def training_history_service(tmp_path_factory) -> TrainingHistoryService:
    return TrainingHistoryService(tmp_path_factory.mktemp("history_session"))


@pytest.fixture(scope="session")
def model_training_service(
    preprocessing_service: PreprocessingService,
    model_storage_service: ModelStorageService,
    training_history_service: TrainingHistoryService,
) -> ModelTrainingService:
    return ModelTrainingService(
        preprocessing_service, model_storage_service, training_history_service
    )


@pytest.fixture(scope="session")
def prediction_service(model_storage_service: ModelStorageService) -> PredictionService:
    return PredictionService(model_storage_service)


@pytest.fixture(scope="session")
def app(
    test_settings: Settings,
    dataset_service: DatasetService,
    preprocessing_service: PreprocessingService,
    model_storage_service: ModelStorageService,
    model_training_service: ModelTrainingService,
    training_history_service: TrainingHistoryService,
    prediction_service: PredictionService,
):
    application = create_app(settings=test_settings)
    application.dependency_overrides[get_dataset_service] = lambda: dataset_service
    application.dependency_overrides[get_preprocessing_service] = lambda: preprocessing_service
    application.dependency_overrides[get_model_storage_service] = lambda: model_storage_service
    application.dependency_overrides[get_model_training_service] = lambda: model_training_service
    application.dependency_overrides[get_training_history_service] = lambda: (
        training_history_service
    )
    application.dependency_overrides[get_prediction_service] = lambda: prediction_service
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
