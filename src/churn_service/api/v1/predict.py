from fastapi import APIRouter

from churn_service.schemas.features import FeatureVectorChurn
from churn_service.services.prediction import PredictionService

router = APIRouter()
_service = PredictionService()


@router.post("/")
def predict(features: FeatureVectorChurn) -> dict:
    return _service.predict(features)
