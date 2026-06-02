from pathlib import Path

import pandas as pd
import pytest

from churn_service.schemas.training import TrainResponse
from churn_service.services.model_storage import ModelStorageService, TrainedModel
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
def trained_model(service: ModelTrainingService) -> TrainedModel:
    return service.train()


@pytest.fixture(scope="module")
def train_response(service: ModelTrainingService) -> TrainResponse:
    return service.train_and_save()


# ---------------------------------------------------------------------------
# train() — pure training, no persistence
# ---------------------------------------------------------------------------


def test_train_returns_trained_model(trained_model: TrainedModel) -> None:
    assert isinstance(trained_model, TrainedModel)


def test_train_accuracy_is_in_valid_range(trained_model: TrainedModel) -> None:
    assert 0.0 <= trained_model.accuracy <= 1.0


def test_train_f1_is_in_valid_range(trained_model: TrainedModel) -> None:
    assert 0.0 <= trained_model.f1 <= 1.0


def test_train_size_matches_split(
    trained_model: TrainedModel, preprocessing: PreprocessingService
) -> None:
    split = preprocessing.prepare_split()
    assert trained_model.train_size == len(split.y_train)


def test_test_size_matches_split(
    trained_model: TrainedModel, preprocessing: PreprocessingService
) -> None:
    split = preprocessing.prepare_split()
    assert trained_model.test_size == len(split.y_test)


def test_train_does_not_persist(preprocessing: PreprocessingService, tmp_path_factory) -> None:
    isolated_storage = ModelStorageService(tmp_path_factory.mktemp("no_persist"))
    service = ModelTrainingService(preprocessing, isolated_storage)
    service.train()
    assert isolated_storage.current is None


def test_training_is_reproducible(
    service: ModelTrainingService, trained_model: TrainedModel
) -> None:
    second = service.train()
    assert second.accuracy == trained_model.accuracy
    assert second.f1 == trained_model.f1


# ---------------------------------------------------------------------------
# train_and_save() — training + persistence + API response
# ---------------------------------------------------------------------------


def test_train_and_save_returns_train_response(train_response: TrainResponse) -> None:
    assert isinstance(train_response, TrainResponse)


def test_train_and_save_persists_model(
    train_response: TrainResponse, storage: ModelStorageService
) -> None:
    # train_response fixture calls train_and_save(), which must populate storage
    assert storage.current is not None


def test_train_and_save_metrics_match_storage(
    train_response: TrainResponse, storage: ModelStorageService
) -> None:
    assert storage.current is not None
    assert train_response.accuracy == storage.current.accuracy
    assert train_response.f1 == storage.current.f1
    assert train_response.train_size == storage.current.train_size
    assert train_response.test_size == storage.current.test_size
