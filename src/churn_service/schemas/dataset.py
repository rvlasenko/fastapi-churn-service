from pydantic import BaseModel

from churn_service.schemas.features import DatasetRowChurn


class ChurnDistribution(BaseModel):
    retained: int
    churned: int


class DatasetInfoResponse(BaseModel):
    row_count: int
    column_count: int
    feature_names: list[str]
    churn_distribution: ChurnDistribution


class DatasetPreviewResponse(BaseModel):
    rows: list[DatasetRowChurn]
    total_returned: int


class SplitChurnDistribution(BaseModel):
    total: int
    retained: int
    churned: int
    churn_rate: float  # churned / total, rounded to 4 decimal places


class DatasetSplitInfoResponse(BaseModel):
    train_size: int
    test_size: int
    test_ratio: float  # actual test_size / total, rounded to 4 decimal places
    numerical_features: list[str]
    categorical_features: list[str]
    train_distribution: SplitChurnDistribution
    test_distribution: SplitChurnDistribution
