from pydantic import BaseModel, Field

from churn_service.schemas.features import FeatureVectorChurn


class PredictItem(BaseModel):
    predicted_class: int
    churn_probability: float
    retained_probability: float


class PredictRequest(BaseModel):
    items: list[FeatureVectorChurn] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: list[PredictItem]
