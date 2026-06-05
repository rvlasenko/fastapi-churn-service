import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from churn_service.api.v1.router import router
from churn_service.core.config import Settings
from churn_service.core.error_handlers import register_error_handlers
from churn_service.core.exceptions import ModelLoadError
from churn_service.core.logging import setup_logging
from churn_service.dependencies import get_settings
from churn_service.services.dataset import DatasetService
from churn_service.services.model_storage import ModelStorageService
from churn_service.services.training_history import TrainingHistoryService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    setup_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Model is loaded here — at startup, before any request is served.
        try:
            storage = ModelStorageService(settings.models_dir)
        except ModelLoadError as exc:
            raise RuntimeError("App startup failed: model file is corrupted") from exc
        app.state.model_storage_service = storage

        # History file is read lazily — missing or corrupted file surfaces at
        # request time, not at startup.
        app.state.training_history_service = TrainingHistoryService(settings.models_dir)

        # Dataset probe: verify dataset availability at startup and expose the
        # result via app.state for the health endpoint. Does not replace the
        # cached get_dataset_service() dependency used by other routes.
        try:
            DatasetService(settings.dataset_path)
            app.state.dataset_loaded = True
            logger.info("Dataset probe succeeded: path=%s", settings.dataset_path)
        except Exception as exc:
            app.state.dataset_loaded = False
            logger.warning("Dataset not available at startup: %s", exc)

        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api/v1")
    register_error_handlers(app)
    return app


app = create_app()
