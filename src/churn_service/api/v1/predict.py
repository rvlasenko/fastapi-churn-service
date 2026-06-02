from fastapi import APIRouter, Depends

from churn_service.dependencies import get_prediction_service
from churn_service.schemas.prediction import PredictRequest, PredictResponse
from churn_service.services.prediction import PredictionService

router = APIRouter()


@router.post("/")
def predict(
    request: PredictRequest,
    service: PredictionService = Depends(get_prediction_service),  # noqa: B008
) -> PredictResponse:
    return service.predict(request.items)
