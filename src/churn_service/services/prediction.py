from churn_service.schemas.features import FeatureVectorChurn
from churn_service.schemas.prediction import PredictionResponse


class PredictionService:
    def predict(self, features: FeatureVectorChurn) -> PredictionResponse:
        # Stub: returns the input features unchanged until the model is trained.
        return PredictionResponse(
            prediction=None,
            churn_probability=None,
            input=features,
        )
