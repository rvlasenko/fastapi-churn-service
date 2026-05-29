from fastapi import APIRouter

from churn_service.api.v1 import dataset, health, predict

router = APIRouter()
router.include_router(health.router, tags=["health"])
router.include_router(predict.router, prefix="/predict", tags=["prediction"])
router.include_router(dataset.router, prefix="/dataset", tags=["dataset"])
