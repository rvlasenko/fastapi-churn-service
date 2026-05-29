from churn_service.schemas.features import FeatureVectorChurn


class PredictionService:
    def predict(self, features: FeatureVectorChurn) -> dict:
        # Stub: returns the input features unchanged until the model is trained.
        return {
            "prediction": None,
            "churn_probability": None,
            "input": features.model_dump(),
        }
