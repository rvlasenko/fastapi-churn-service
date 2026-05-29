import pytest
from pydantic import ValidationError

from churn_service.schemas.features import DatasetRowChurn, FeatureVectorChurn

VALID_DATA = {
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


def test_valid_feature_vector_passes() -> None:
    obj = FeatureVectorChurn(**VALID_DATA)
    assert obj.region == "europe"
    assert obj.monthly_fee == 29.99


def test_monthly_fee_zero_fails() -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn(**{**VALID_DATA, "monthly_fee": 0})


def test_monthly_fee_negative_fails() -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn(**{**VALID_DATA, "monthly_fee": -5})


def test_usage_hours_negative_fails() -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn(**{**VALID_DATA, "usage_hours": -1})


def test_usage_hours_over_monthly_max_fails() -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn(**{**VALID_DATA, "usage_hours": 800})


def test_usage_hours_zero_passes() -> None:
    obj = FeatureVectorChurn(**{**VALID_DATA, "usage_hours": 0})
    assert obj.usage_hours == 0


def test_invalid_region_fails() -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn(**{**VALID_DATA, "region": "narnia"})


def test_invalid_device_type_fails() -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn(**{**VALID_DATA, "device_type": "smartwatch"})


def test_invalid_payment_method_fails() -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn(**{**VALID_DATA, "payment_method": "bitcoin"})


def test_autopay_enabled_two_fails() -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn(**{**VALID_DATA, "autopay_enabled": 2})


def test_autopay_enabled_zero_passes() -> None:
    obj = FeatureVectorChurn(**{**VALID_DATA, "autopay_enabled": 0})
    assert obj.autopay_enabled == 0


def test_dataset_row_with_churn_label_passes() -> None:
    obj = DatasetRowChurn(**{**VALID_DATA, "churn": 1})
    assert obj.churn == 1


def test_dataset_row_churn_two_fails() -> None:
    with pytest.raises(ValidationError):
        DatasetRowChurn(**{**VALID_DATA, "churn": 2})


def test_dataset_row_churn_negative_fails() -> None:
    with pytest.raises(ValidationError):
        DatasetRowChurn(**{**VALID_DATA, "churn": -1})
