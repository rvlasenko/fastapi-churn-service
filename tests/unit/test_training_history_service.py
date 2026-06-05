import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from churn_service.core.exceptions import HistoryLoadError, HistoryWriteError
from churn_service.schemas.history import TrainingRecord
from churn_service.services.training_history import TrainingHistoryService

_SAMPLE_RECORD = TrainingRecord(
    trained_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
    model_type="logreg",
    hyperparameters={},
    accuracy=0.85,
    f1=0.82,
    roc_auc=0.91,
    train_size=6400,
    test_size=1600,
)

_SAMPLE_RF_RECORD = TrainingRecord(
    trained_at=datetime(2026, 6, 2, 12, 0, 0, tzinfo=UTC),
    model_type="random_forest",
    hyperparameters={"n_estimators": 100},
    accuracy=0.87,
    f1=0.84,
    roc_auc=0.93,
    train_size=6400,
    test_size=1600,
)

_SAMPLE_NULL_ROC = TrainingRecord(
    trained_at=datetime(2026, 6, 3, 12, 0, 0, tzinfo=UTC),
    model_type="logreg",
    hyperparameters={},
    accuracy=1.0,
    f1=1.0,
    roc_auc=None,
    train_size=6400,
    test_size=1600,
)


@pytest.fixture
def history_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def service(history_dir: Path) -> TrainingHistoryService:
    return TrainingHistoryService(history_dir)


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------


def test_append_creates_file_when_missing(
    service: TrainingHistoryService, history_dir: Path
) -> None:
    history_file = history_dir / "training_history.json"
    assert not history_file.exists()
    service.append(_SAMPLE_RECORD)
    assert history_file.exists()


def test_append_stores_one_record(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_RECORD)
    records = service.load(limit=100)
    assert len(records) == 1


def test_append_stores_second_record(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_RECORD)
    service.append(_SAMPLE_RF_RECORD)
    records = service.load(limit=100)
    assert len(records) == 2


def test_append_newest_first_ordering(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_RECORD)
    service.append(_SAMPLE_RF_RECORD)
    records = service.load(limit=100)
    # Most recently appended comes first
    assert records[0].model_type == _SAMPLE_RF_RECORD.model_type
    assert records[1].model_type == _SAMPLE_RECORD.model_type


def test_append_raises_history_write_error_on_io_failure(
    service: TrainingHistoryService,
) -> None:
    with patch.object(service, "_write_raw", side_effect=OSError("disk full")):
        with pytest.raises((HistoryWriteError, OSError)):
            service.append(_SAMPLE_RECORD)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def test_load_returns_empty_list_when_file_missing(service: TrainingHistoryService) -> None:
    assert service.load() == []


def test_load_returns_all_records(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_RECORD)
    service.append(_SAMPLE_RF_RECORD)
    records = service.load(limit=100)
    assert len(records) == 2


def test_load_applies_limit(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_RECORD)
    service.append(_SAMPLE_RF_RECORD)
    records = service.load(limit=1)
    assert len(records) == 1


def test_load_limit_returns_most_recent(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_RECORD)
    service.append(_SAMPLE_RF_RECORD)
    records = service.load(limit=1)
    assert records[0].model_type == _SAMPLE_RF_RECORD.model_type


def test_load_filters_by_model_type(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_RECORD)
    service.append(_SAMPLE_RF_RECORD)
    logreg_records = service.load(model_type="logreg", limit=100)
    assert all(r.model_type == "logreg" for r in logreg_records)
    assert len(logreg_records) == 1


def test_load_filter_returns_empty_for_unmatched_type(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_RECORD)
    records = service.load(model_type="random_forest", limit=100)
    assert records == []


def test_load_raises_history_load_error_on_corrupted_json(
    service: TrainingHistoryService, history_dir: Path
) -> None:
    (history_dir / "training_history.json").write_text("not valid json", encoding="utf-8")
    with pytest.raises(HistoryLoadError):
        service.load()


def test_load_raises_history_load_error_when_file_is_not_array(
    service: TrainingHistoryService, history_dir: Path
) -> None:
    (history_dir / "training_history.json").write_text('{"key": "value"}', encoding="utf-8")
    with pytest.raises(HistoryLoadError):
        service.load()


def test_load_raises_history_load_error_on_invalid_record_schema(
    service: TrainingHistoryService, history_dir: Path
) -> None:
    bad_data = json.dumps([{"trained_at": "not-a-date", "model_type": 123}])
    (history_dir / "training_history.json").write_text(bad_data, encoding="utf-8")
    with pytest.raises(HistoryLoadError):
        service.load()


# ---------------------------------------------------------------------------
# roc_auc handling
# ---------------------------------------------------------------------------


def test_roc_auc_float_stored_and_retrieved(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_RECORD)
    records = service.load(limit=1)
    assert records[0].roc_auc == _SAMPLE_RECORD.roc_auc


def test_roc_auc_none_stored_and_retrieved(service: TrainingHistoryService) -> None:
    service.append(_SAMPLE_NULL_ROC)
    records = service.load(limit=1)
    assert records[0].roc_auc is None


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_append_logs_history_record(
    caplog: pytest.LogCaptureFixture, service: TrainingHistoryService
) -> None:
    with caplog.at_level(logging.INFO, logger="churn_service.services.training_history"):
        service.append(_SAMPLE_RECORD)

    messages = [r.message for r in caplog.records]
    assert any("History record appended" in m for m in messages)
    assert any(_SAMPLE_RECORD.model_type in m for m in messages)
