class ChurnServiceError(Exception):
    """Base exception for all domain errors."""


class DatasetValidationError(ChurnServiceError):
    """Raised when the dataset does not meet the expected schema."""


class ModelNotFoundError(ChurnServiceError):
    """Raised when model file does not exist on disk."""


class ModelLoadError(ChurnServiceError):
    """Raised when model file exists but cannot be deserialized."""
