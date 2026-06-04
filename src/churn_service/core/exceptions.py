class ChurnServiceError(Exception):
    """Base exception for all domain errors."""


class DatasetError(ChurnServiceError):
    """Raised when the dataset is missing or unreadable."""


class DatasetValidationError(DatasetError):
    """Raised when the dataset does not meet the expected schema."""


class ModelError(ChurnServiceError):
    """Base exception for model-related failures."""


class ModelNotTrainedError(ModelError):
    """Raised when no trained model is available."""


class ModelLoadError(ModelError):
    """Raised when a model file exists but cannot be deserialized."""
