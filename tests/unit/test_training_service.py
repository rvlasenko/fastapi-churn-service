from pathlib import Path

import pandas as pd
import pytest

from churn_service.schemas.training import TrainResponse
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.preprocessing import PreprocessingService
from churn_service.services.training import ModelTrainingService

DATASET_PATH = Path("data/churn_dataset.csv")


@pytest.fixture(scope="module")
def preprocessing() -> PreprocessingService:
    return PreprocessingService(pd.read_csv(DATASET_PATH))


@pytest.fixture(scope="module")
def storage(tmp_path_factory) -> ModelStorageService:
    return ModelStorageService(tmp_path_factory.mktemp("models_training_unit"))


@pytest.fixture(scope="module")
def service(
    preprocessing: PreprocessingService, storage: ModelStorageService
) -> ModelTrainingService:
    return ModelTrainingService(preprocessing, storage)


@pytest.fixture(scope="module")
def result(service: ModelTrainingService) -> TrainResponse:
    return service.train()


def test_train_returns_train_response(result: TrainResponse) -> None:
    assert isinstance(result, TrainResponse)


def test_accuracy_is_in_valid_range(result: TrainResponse) -> None:
    assert 0.0 <= result.accuracy <= 1.0


def test_f1_is_in_valid_range(result: TrainResponse) -> None:
    assert 0.0 <= result.f1 <= 1.0


def test_train_size_matches_split(
    result: TrainResponse, preprocessing: PreprocessingService
) -> None:
    split = preprocessing.prepare_split()
    assert result.train_size == len(split.y_train)


def test_test_size_matches_split(
    result: TrainResponse, preprocessing: PreprocessingService
) -> None:
    split = preprocessing.prepare_split()
    assert result.test_size == len(split.y_test)


def test_training_is_reproducible(service: ModelTrainingService, result: TrainResponse) -> None:
    second = service.train()
    assert second.accuracy == result.accuracy
    assert second.f1 == result.f1
