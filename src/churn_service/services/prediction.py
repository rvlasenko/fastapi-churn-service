import pandas as pd
from fastapi import HTTPException

from churn_service.schemas.features import FeatureVectorChurn
from churn_service.schemas.prediction import PredictItem, PredictResponse
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.preprocessing import FEATURE_COLUMNS


class PredictionService:
    def __init__(self, storage: ModelStorageService) -> None:
        self._storage = storage

    def predict(self, items: list[FeatureVectorChurn]) -> PredictResponse:
        model = self._storage.current
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="No trained model available. Train the model first via POST /api/v1/model/train",
            )

        df = pd.DataFrame(
            [item.model_dump() for item in items],
            columns=FEATURE_COLUMNS,
        )

        predicted_classes = model.pipeline.predict(df)
        probabilities = model.pipeline.predict_proba(df)  # shape (n, 2): col0=retained, col1=churn

        return PredictResponse(
            predictions=[
                PredictItem(
                    predicted_class=int(predicted_classes[i]),
                    churn_probability=float(probabilities[i, 1]),
                    retained_probability=float(probabilities[i, 0]),
                )
                for i in range(len(items))
            ]
        )
