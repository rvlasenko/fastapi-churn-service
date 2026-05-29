class ChurnServiceError(Exception):
    """Base exception for all domain errors."""


class DatasetValidationError(ChurnServiceError):
    """Raised when the dataset does not meet the expected schema."""
