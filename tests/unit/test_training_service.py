import logging
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from churn_service.core.exceptions import HistoryWriteError
from churn_service.schemas.training import TrainResponse
from churn_service.services.model_storage import ModelStorageService, TrainedModel
from churn_service.services.preprocessing import PreprocessingService
from churn_service.services.training import ModelTrainingService
from churn_service.services.training_history import TrainingHistoryService

DATASET_PATH = Path("data/churn_dataset.csv")


@pytest.fixture(scope="module")
def preprocessing() -> PreprocessingService:
    return PreprocessingService(pd.read_csv(DATASET_PATH))


@pytest.fixture(scope="module")
def storage(tmp_path_factory) -> ModelStorageService:
    return ModelStorageService(tmp_path_factory.mktemp("models_training_unit"))


@pytest.fixture(scope="module")
def history_service(tmp_path_factory) -> TrainingHistoryService:
    return TrainingHistoryService(tmp_path_factory.mktemp("history_training_unit"))


@pytest.fixture(scope="module")
def service(
    preprocessing: PreprocessingService,
    storage: ModelStorageService,
    history_service: TrainingHistoryService,
) -> ModelTrainingService:
    return ModelTrainingService(preprocessing, storage, history_service)


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


def test_train_roc_auc_is_in_valid_range(trained_model: TrainedModel) -> None:
    assert trained_model.roc_auc is not None
    assert 0.0 <= trained_model.roc_auc <= 1.0


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
    tmpdir = tmp_path_factory.mktemp("no_persist")
    isolated_storage = ModelStorageService(tmpdir)
    isolated_history = TrainingHistoryService(tmpdir)
    service = ModelTrainingService(preprocessing, isolated_storage, isolated_history)
    service.train()
    assert isolated_storage.current is None


def test_training_is_reproducible(
    service: ModelTrainingService, trained_model: TrainedModel
) -> None:
    second = service.train()
    assert second.accuracy == trained_model.accuracy
    assert second.f1 == trained_model.f1
    assert second.roc_auc == trained_model.roc_auc


# ---------------------------------------------------------------------------
# train_and_save() — training + persistence + API response
# ---------------------------------------------------------------------------


def test_train_and_save_returns_train_response(train_response: TrainResponse) -> None:
    assert isinstance(train_response, TrainResponse)


def test_train_and_save_response_includes_roc_auc(train_response: TrainResponse) -> None:
    assert train_response.roc_auc is not None
    assert 0.0 <= train_response.roc_auc <= 1.0


def test_train_and_save_persists_model(
    train_response: TrainResponse, storage: ModelStorageService
) -> None:
    assert storage.current is not None


def test_train_and_save_metrics_match_storage(
    train_response: TrainResponse, storage: ModelStorageService
) -> None:
    assert storage.current is not None
    assert train_response.accuracy == storage.current.accuracy
    assert train_response.f1 == storage.current.f1
    assert train_response.roc_auc == storage.current.roc_auc
    assert train_response.train_size == storage.current.train_size
    assert train_response.test_size == storage.current.test_size


def test_train_and_save_appends_one_history_record(
    preprocessing: PreprocessingService, tmp_path_factory
) -> None:
    tmpdir = tmp_path_factory.mktemp("history_append_test")
    isolated_storage = ModelStorageService(tmpdir)
    isolated_history = TrainingHistoryService(tmpdir)
    svc = ModelTrainingService(preprocessing, isolated_storage, isolated_history)

    svc.train_and_save()
    assert len(isolated_history.load(limit=100)) == 1

    svc.train_and_save()
    assert len(isolated_history.load(limit=100)) == 2


def test_train_and_save_does_not_append_history_when_storage_fails(
    preprocessing: PreprocessingService, tmp_path_factory
) -> None:
    tmpdir = tmp_path_factory.mktemp("history_no_append_on_fail")
    broken_storage = MagicMock(spec=ModelStorageService)
    broken_storage.save.side_effect = RuntimeError("disk full")
    isolated_history = TrainingHistoryService(tmpdir)
    svc = ModelTrainingService(preprocessing, broken_storage, isolated_history)

    with pytest.raises(RuntimeError):
        svc.train_and_save()

    assert isolated_history.load(limit=100) == []


def test_train_and_save_propagates_history_write_error(
    preprocessing: PreprocessingService, tmp_path_factory
) -> None:
    tmpdir = tmp_path_factory.mktemp("history_write_error")
    isolated_storage = ModelStorageService(tmpdir)
    broken_history = MagicMock(spec=TrainingHistoryService)
    broken_history.append.side_effect = HistoryWriteError("write failed")
    svc = ModelTrainingService(preprocessing, isolated_storage, broken_history)

    with pytest.raises(HistoryWriteError):
        svc.train_and_save()


def test_history_append_failure_logs_append_failed_not_training_failed(
    caplog: pytest.LogCaptureFixture,
    preprocessing: PreprocessingService,
    tmp_path_factory,
) -> None:
    tmpdir = tmp_path_factory.mktemp("history_fail_log")
    broken_history = MagicMock(spec=TrainingHistoryService)
    broken_history.append.side_effect = HistoryWriteError("disk full")
    svc = ModelTrainingService(preprocessing, ModelStorageService(tmpdir), broken_history)

    with caplog.at_level(logging.DEBUG, logger="churn_service.services.training"):
        with pytest.raises(HistoryWriteError):
            svc.train_and_save()

    messages = [r.message for r in caplog.records]
    assert any("History append failed" in m for m in messages)
    assert not any("Training failed" in m for m in messages)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_train_and_save_logs_started_and_succeeded(
    caplog: pytest.LogCaptureFixture,
    preprocessing: PreprocessingService,
    tmp_path_factory,
) -> None:
    tmpdir = tmp_path_factory.mktemp("logging_started")
    svc = ModelTrainingService(
        preprocessing,
        ModelStorageService(tmpdir),
        TrainingHistoryService(tmpdir),
    )
    with caplog.at_level(logging.INFO, logger="churn_service.services.training"):
        svc.train_and_save()

    messages = [r.message for r in caplog.records]
    assert any("Training started" in m for m in messages)
    assert any("Training succeeded" in m for m in messages)


def test_train_and_save_does_not_log_raw_payload(
    caplog: pytest.LogCaptureFixture,
    preprocessing: PreprocessingService,
    tmp_path_factory,
) -> None:
    tmpdir = tmp_path_factory.mktemp("logging_payload")
    svc = ModelTrainingService(
        preprocessing,
        ModelStorageService(tmpdir),
        TrainingHistoryService(tmpdir),
    )
    with caplog.at_level(logging.DEBUG, logger="churn_service.services.training"):
        svc.train_and_save()

    full_log = " ".join(r.message for r in caplog.records)
    assert "X_train" not in full_log
    assert "DataFrame" not in full_log
