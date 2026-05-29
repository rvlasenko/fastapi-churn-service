from functools import lru_cache

from fastapi import HTTPException

from churn_service.core.config import Settings
from churn_service.core.exceptions import DatasetValidationError
from churn_service.services.dataset import DatasetService
from churn_service.services.preprocessing import PreprocessingService


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_dataset_service() -> DatasetService:
    try:
        return DatasetService(get_settings().dataset_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail="Dataset file not found") from e
    except DatasetValidationError as e:
        raise HTTPException(status_code=503, detail=f"Dataset schema invalid: {e}") from e


@lru_cache
def get_preprocessing_service() -> PreprocessingService:
    # Cached for the process lifetime.
    # If the dataset CSV changes on disk, restart the server to pick up changes.
    try:
        service = PreprocessingService(get_dataset_service().get_dataframe())
        service.prepare_split()  # validates NaN and caches the split eagerly
        return service
    except DatasetValidationError as e:
        raise HTTPException(status_code=503, detail=f"Dataset preprocessing failed: {e}") from e
