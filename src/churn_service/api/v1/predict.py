from fastapi import APIRouter, Depends

from churn_service.dependencies import get_prediction_service
from churn_service.schemas.features import FeatureVectorChurn
from churn_service.schemas.prediction import PredictionResponse
from churn_service.services.prediction import PredictionService

router = APIRouter()


@router.post("/")
def predict(
    features: FeatureVectorChurn,
    service: PredictionService = Depends(get_prediction_service),  # noqa: B008
) -> PredictionResponse:
    return service.predict(features)
