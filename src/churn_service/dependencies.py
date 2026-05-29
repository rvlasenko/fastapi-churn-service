from functools import lru_cache

from fastapi import HTTPException

from churn_service.core.config import Settings
from churn_service.core.exceptions import DatasetValidationError
from churn_service.services.dataset import DatasetService


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
