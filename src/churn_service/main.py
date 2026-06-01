from contextlib import asynccontextmanager

from fastapi import FastAPI

from churn_service.api.v1.router import router
from churn_service.core.config import Settings
from churn_service.core.exceptions import ModelLoadError
from churn_service.dependencies import get_settings
from churn_service.services.model_storage import ModelStorageService


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Model is loaded here — at startup, before any request is served.
        try:
            storage = ModelStorageService(settings.models_dir)
        except ModelLoadError as exc:
            raise RuntimeError("App startup failed: model file is corrupted") from exc
        app.state.model_storage_service = storage
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
