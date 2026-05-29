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
