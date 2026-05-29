from fastapi import APIRouter, Depends, Query

from churn_service.dependencies import get_dataset_service
from churn_service.schemas.dataset import DatasetInfoResponse, DatasetPreviewResponse
from churn_service.services.dataset import DatasetService

router = APIRouter()


@router.get("/preview")
def preview(
    n: int = Query(default=10, ge=1, le=500, description="Number of rows to return"),
    service: DatasetService = Depends(get_dataset_service),  # noqa: B008
) -> DatasetPreviewResponse:
    rows = service.get_preview(n)
    return DatasetPreviewResponse(rows=rows, total_returned=len(rows))


@router.get("/info")
def info(service: DatasetService = Depends(get_dataset_service)) -> DatasetInfoResponse:  # noqa: B008
    return service.get_info()
