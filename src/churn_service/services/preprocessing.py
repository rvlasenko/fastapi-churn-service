from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_service.core.exceptions import DatasetValidationError
from churn_service.schemas.dataset import DatasetSplitInfoResponse, SplitChurnDistribution

TARGET_COLUMN: str = "churn"

NUMERICAL_FEATURES: list[str] = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",  # binary 0/1 int — treated as numerical in sklearn pipelines
]

CATEGORICAL_FEATURES: list[str] = [
    "region",
    "device_type",
    "payment_method",
]

FEATURE_COLUMNS: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42  # fixed seed — split is deterministic for the process lifetime


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class PreprocessingService:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._split: DataSplit | None = None

    def prepare_split(self) -> DataSplit:
        """Validate the DataFrame and compute the train/test split.

        Idempotent — result is cached on first call.
        The cache lives for the process lifetime; restart required if CSV changes.
        """
        if self._split is None:
            self._split = self._compute_split()
        return self._split

    def _compute_split(self) -> DataSplit:
        nan_cols = [col for col in self._df.columns if bool(self._df[col].isna().any())]
        if nan_cols:
            # Real missing-value imputation (median fill, mode fill, etc.) will belong
            # in the sklearn preprocessing pipeline in a future day.
            raise DatasetValidationError(f"Dataset has missing values in columns: {nan_cols}")

        X = self._df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]  # noqa: N806
        y = self._df[TARGET_COLUMN]

        split = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        X_train = cast(pd.DataFrame, split[0])  # noqa: N806
        X_test = cast(pd.DataFrame, split[1])  # noqa: N806
        y_train = cast(pd.Series, split[2])  # noqa: N806
        y_test = cast(pd.Series, split[3])  # noqa: N806

        return DataSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

    def get_split_info(self) -> DatasetSplitInfoResponse:
        split = self.prepare_split()
        train_total = len(split.y_train)
        test_total = len(split.y_test)
        train_churned = int(split.y_train.sum())
        test_churned = int(split.y_test.sum())
        return DatasetSplitInfoResponse(
            train_size=train_total,
            test_size=test_total,
            test_ratio=round(test_total / (train_total + test_total), 4),
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            train_distribution=SplitChurnDistribution(
                total=train_total,
                retained=train_total - train_churned,
                churned=train_churned,
                churn_rate=round(train_churned / train_total, 4),
            ),
            test_distribution=SplitChurnDistribution(
                total=test_total,
                retained=test_total - test_churned,
                churned=test_churned,
                churn_rate=round(test_churned / test_total, 4),
            ),
        )
