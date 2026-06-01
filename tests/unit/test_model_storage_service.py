from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_service.core.exceptions import ModelLoadError, ModelNotFoundError
from churn_service.services.model_storage import ModelStorageService, TrainedModel


@pytest.fixture
def minimal_pipeline() -> Pipeline:
    pipeline = Pipeline([("classifier", LogisticRegression(random_state=42))])
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})  # noqa: N806
    y = pd.Series([0, 1, 0, 1])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def sample_trained_model(minimal_pipeline: Pipeline) -> TrainedModel:
    return TrainedModel(
        pipeline=minimal_pipeline,
        trained_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        model_type="LogisticRegression",
        accuracy=0.75,
        f1=0.8,
        train_size=100,
        test_size=25,
    )


def test_initial_state_is_none_when_no_file(tmp_path: Path) -> None:
    storage = ModelStorageService(tmp_path)
    assert storage.current is None


def test_save_creates_model_file(tmp_path: Path, sample_trained_model: TrainedModel) -> None:
    storage = ModelStorageService(tmp_path)
    storage.save(sample_trained_model)
    assert (tmp_path / "churn_model.joblib").exists()


def test_save_updates_current_in_memory(tmp_path: Path, sample_trained_model: TrainedModel) -> None:
    storage = ModelStorageService(tmp_path)
    assert storage.current is None
    storage.save(sample_trained_model)
    assert storage.current is sample_trained_model


def test_load_returns_saved_model(tmp_path: Path, sample_trained_model: TrainedModel) -> None:
    storage = ModelStorageService(tmp_path)
    storage.save(sample_trained_model)
    loaded = storage.load()
    assert loaded.model_type == sample_trained_model.model_type
    assert loaded.accuracy == sample_trained_model.accuracy
    assert loaded.f1 == sample_trained_model.f1
    assert loaded.train_size == sample_trained_model.train_size
    assert loaded.test_size == sample_trained_model.test_size


def test_load_raises_model_not_found_when_no_file(tmp_path: Path) -> None:
    storage = ModelStorageService(tmp_path)
    with pytest.raises(ModelNotFoundError):
        storage.load()


def test_load_raises_model_load_error_on_corrupted_file(tmp_path: Path) -> None:
    storage = ModelStorageService(tmp_path)
    (tmp_path / "churn_model.joblib").write_bytes(b"not a joblib file")
    with pytest.raises(ModelLoadError):
        storage.load()


def test_exists_returns_false_when_no_file(tmp_path: Path) -> None:
    storage = ModelStorageService(tmp_path)
    assert storage.exists() is False


def test_exists_returns_true_after_save(tmp_path: Path, sample_trained_model: TrainedModel) -> None:
    storage = ModelStorageService(tmp_path)
    storage.save(sample_trained_model)
    assert storage.exists() is True


def test_corrupted_file_raises_model_load_error_on_init(tmp_path: Path) -> None:
    (tmp_path / "churn_model.joblib").write_bytes(b"not a joblib file")
    with pytest.raises(ModelLoadError):
        ModelStorageService(tmp_path)


def test_fresh_instance_loads_existing_model_from_disk(
    tmp_path: Path, sample_trained_model: TrainedModel
) -> None:
    storage_one = ModelStorageService(tmp_path)
    storage_one.save(sample_trained_model)

    storage_two = ModelStorageService(tmp_path)
    assert storage_two.current is not None
    assert storage_two.current.model_type == sample_trained_model.model_type
    assert storage_two.current.accuracy == sample_trained_model.accuracy
    assert storage_two.current.trained_at == sample_trained_model.trained_at
