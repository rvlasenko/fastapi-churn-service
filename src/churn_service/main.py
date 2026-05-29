from fastapi import FastAPI

from churn_service.api.v1.router import router
from churn_service.core.config import Settings
from churn_service.dependencies import get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
