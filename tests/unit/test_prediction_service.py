import logging
from unittest.mock import MagicMock

import pytest

from churn_service.core.exceptions import ModelNotTrainedError
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.prediction import PredictionService


def test_predict_logs_batch_size_when_model_absent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    storage = MagicMock(spec=ModelStorageService)
    storage.current = None
    service = PredictionService(storage)

    with caplog.at_level(logging.DEBUG, logger="churn_service.services.prediction"):
        with pytest.raises(ModelNotTrainedError):
            service.predict([])

    messages = [r.message for r in caplog.records]
    assert any("batch_size=0" in m for m in messages)


def test_predict_warns_when_model_not_trained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    storage = MagicMock(spec=ModelStorageService)
    storage.current = None
    service = PredictionService(storage)

    with caplog.at_level(logging.WARNING, logger="churn_service.services.prediction"):
        with pytest.raises(ModelNotTrainedError):
            service.predict([])

    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_predict_does_not_log_raw_items(
    caplog: pytest.LogCaptureFixture,
) -> None:
    storage = MagicMock(spec=ModelStorageService)
    storage.current = None
    service = PredictionService(storage)

    with caplog.at_level(logging.DEBUG, logger="churn_service.services.prediction"):
        with pytest.raises(ModelNotTrainedError):
            service.predict([])

    full_log = " ".join(r.message for r in caplog.records)
    assert "monthly_fee" not in full_log
    assert "model_dump" not in full_log
