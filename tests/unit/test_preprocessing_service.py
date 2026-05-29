from pathlib import Path

import pandas as pd
import pytest

from churn_service.core.exceptions import DatasetValidationError
from churn_service.services.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    DataSplit,
    PreprocessingService,
)

DATASET_PATH = Path("data/churn_dataset.csv")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


@pytest.fixture(scope="module")
def service(df: pd.DataFrame) -> PreprocessingService:
    return PreprocessingService(df)


@pytest.fixture(scope="module")
def split(service: PreprocessingService) -> DataSplit:
    return service.prepare_split()


def test_target_excluded_from_feature_matrix(split: DataSplit) -> None:
    assert TARGET_COLUMN not in split.X_train.columns
    assert TARGET_COLUMN not in split.X_test.columns


def test_train_test_sizes_sum_to_total(df: pd.DataFrame, split: DataSplit) -> None:
    assert len(split.X_train) + len(split.X_test) == len(df)


def test_test_split_ratio(df: pd.DataFrame, split: DataSplit) -> None:
    actual_ratio = len(split.X_test) / len(df)
    assert abs(actual_ratio - 0.2) < 0.01


def test_stratification_preserves_churn_rate(split: DataSplit) -> None:
    assert abs(split.y_train.mean() - split.y_test.mean()) < 0.01


def test_split_is_deterministic(df: pd.DataFrame, split: DataSplit) -> None:
    second = PreprocessingService(df).prepare_split()
    assert list(split.X_train.index) == list(second.X_train.index)
    assert list(split.X_test.index) == list(second.X_test.index)


def test_prepare_split_is_idempotent(service: PreprocessingService) -> None:
    first = service.prepare_split()
    second = service.prepare_split()
    assert first is second


def test_nan_raises_dataset_validation_error(df: pd.DataFrame) -> None:
    df_with_nan = df.copy()
    df_with_nan.loc[0, "monthly_fee"] = float("nan")
    with pytest.raises(DatasetValidationError):
        PreprocessingService(df_with_nan).prepare_split()


def test_numerical_features_in_feature_matrix(split: DataSplit) -> None:
    for col in NUMERICAL_FEATURES:
        assert col in split.X_train.columns


def test_categorical_features_in_feature_matrix(split: DataSplit) -> None:
    for col in CATEGORICAL_FEATURES:
        assert col in split.X_train.columns


def test_y_is_binary(split: DataSplit) -> None:
    assert set(split.y_train.unique()) <= {0, 1}
    assert set(split.y_test.unique()) <= {0, 1}
