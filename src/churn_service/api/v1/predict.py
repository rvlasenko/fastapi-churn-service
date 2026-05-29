from fastapi import APIRouter

from churn_service.schemas.features import FeatureVectorChurn
from churn_service.schemas.prediction import PredictionResponse
from churn_service.services.prediction import PredictionService

router = APIRouter()
_service = PredictionService()


@router.post("/")
def predict(features: FeatureVectorChurn) -> PredictionResponse:
    return _service.predict(features)
