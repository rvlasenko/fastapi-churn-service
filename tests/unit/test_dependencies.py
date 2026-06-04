from unittest.mock import MagicMock

import pandas as pd
import pytest

import churn_service.dependencies as deps
from churn_service.core.exceptions import DatasetValidationError


@pytest.fixture(autouse=True)
def clear_cache():
    deps.get_preprocessing_service.cache_clear()
    yield
    deps.get_preprocessing_service.cache_clear()


def test_get_preprocessing_service_raises_dataset_validation_error_on_nan_data(
    monkeypatch,
) -> None:
    df_with_nan = pd.DataFrame(
        {
            "monthly_fee": [float("nan"), 10.0],
            "usage_hours": [100.0, 50.0],
            "support_requests": [1, 0],
            "account_age_months": [12, 6],
            "failed_payments": [0, 1],
            "autopay_enabled": [1, 0],
            "region": ["europe", "asia"],
            "device_type": ["mobile", "desktop"],
            "payment_method": ["card", "paypal"],
            "churn": [0, 1],
        }
    )

    mock_ds = MagicMock()
    mock_ds.get_dataframe.return_value = df_with_nan
    monkeypatch.setattr(deps, "get_dataset_service", lambda: mock_ds)

    with pytest.raises(DatasetValidationError) as exc_info:
        deps.get_preprocessing_service()

    assert "missing values" in str(exc_info.value)
