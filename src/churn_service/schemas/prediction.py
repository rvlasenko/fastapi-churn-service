from pydantic import BaseModel

from churn_service.schemas.features import FeatureVectorChurn


class PredictionResponse(BaseModel):
    prediction: int | None
    churn_probability: float | None
    input: FeatureVectorChurn
