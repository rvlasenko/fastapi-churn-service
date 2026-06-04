import logging
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from churn_service.core.exceptions import (
    ChurnServiceError,
    DatasetError,
    DatasetValidationError,
    HistoryLoadError,
    HistoryWriteError,
    ModelError,
    ModelLoadError,
    ModelNotTrainedError,
)

logger = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    DATASET_NOT_AVAILABLE = "dataset_not_available"
    DATASET_VALIDATION_FAILED = "dataset_validation_failed"
    MODEL_NOT_TRAINED = "model_not_trained"
    MODEL_LOAD_FAILED = "model_load_failed"
    HISTORY_LOAD_FAILED = "history_load_failed"
    HISTORY_WRITE_FAILED = "history_write_failed"
    INTERNAL_ERROR = "internal_error"


_HTTP_STATUS_TO_CODE: dict[int, ErrorCode] = {
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
}


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def _error_response(status: int, code: ErrorCode, message: str, details: Any = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def _sanitize_pydantic_errors(errors: list[Any]) -> list[dict]:
    # Pydantic v2 puts the original exception instance in ctx["error"].
    # That value is not JSON-serializable, so convert it to a string.
    result = []
    for err in errors:
        cleaned = {k: v for k, v in err.items() if k != "url"}
        if "ctx" in cleaned:
            cleaned["ctx"] = {
                k: str(v) if isinstance(v, Exception) else v
                for k, v in cleaned["ctx"].items()
            }
        result.append(cleaned)
    return result


async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = _sanitize_pydantic_errors(list(exc.errors()))
    return _error_response(422, ErrorCode.VALIDATION_ERROR, "Request validation failed", details)


async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    fallback = ErrorCode.VALIDATION_ERROR if exc.status_code < 500 else ErrorCode.INTERNAL_ERROR
    code = _HTTP_STATUS_TO_CODE.get(exc.status_code, fallback)
    return _error_response(exc.status_code, code, str(exc.detail))


async def handle_dataset_validation_error(
    request: Request, exc: DatasetValidationError
) -> JSONResponse:
    return _error_response(503, ErrorCode.DATASET_VALIDATION_FAILED, str(exc))


async def handle_dataset_error(request: Request, exc: DatasetError) -> JSONResponse:
    return _error_response(503, ErrorCode.DATASET_NOT_AVAILABLE, str(exc))


async def handle_model_not_trained_error(
    request: Request, exc: ModelNotTrainedError
) -> JSONResponse:
    return _error_response(503, ErrorCode.MODEL_NOT_TRAINED, str(exc))


async def handle_model_load_error(request: Request, exc: ModelLoadError) -> JSONResponse:
    return _error_response(503, ErrorCode.MODEL_LOAD_FAILED, str(exc))


async def handle_model_error(request: Request, exc: ModelError) -> JSONResponse:
    logger.exception("Unclassified model error")
    return _error_response(503, ErrorCode.INTERNAL_ERROR, "An internal error occurred")


async def handle_history_load_error(request: Request, exc: HistoryLoadError) -> JSONResponse:
    return _error_response(503, ErrorCode.HISTORY_LOAD_FAILED, str(exc))


async def handle_history_write_error(request: Request, exc: HistoryWriteError) -> JSONResponse:
    return _error_response(503, ErrorCode.HISTORY_WRITE_FAILED, str(exc))


async def handle_churn_service_error(request: Request, exc: ChurnServiceError) -> JSONResponse:
    logger.exception("Unclassified domain error")
    return _error_response(500, ErrorCode.INTERNAL_ERROR, "An internal error occurred")


async def handle_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    return _error_response(500, ErrorCode.INTERNAL_ERROR, "An internal error occurred")


def register_error_handlers(app: FastAPI) -> None:
    # Domain exceptions — specific subclasses before parent classes so Starlette's
    # MRO walk reaches the most precise handler first.
    #
    # type: ignore[arg-type] comments below are required because Starlette's type
    # stubs define ExceptionHandler as (Request, Exception) -> Response, but our
    # handlers accept specific exception subclasses. This is correct at runtime.
    app.add_exception_handler(DatasetValidationError, handle_dataset_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(DatasetError, handle_dataset_error)  # type: ignore[arg-type]
    app.add_exception_handler(ModelNotTrainedError, handle_model_not_trained_error)  # type: ignore[arg-type]
    app.add_exception_handler(ModelLoadError, handle_model_load_error)  # type: ignore[arg-type]
    app.add_exception_handler(ModelError, handle_model_error)  # type: ignore[arg-type]
    app.add_exception_handler(HistoryLoadError, handle_history_load_error)  # type: ignore[arg-type]
    app.add_exception_handler(HistoryWriteError, handle_history_write_error)  # type: ignore[arg-type]
    app.add_exception_handler(ChurnServiceError, handle_churn_service_error)  # type: ignore[arg-type]
    # HTTP-layer and broad fallback — override FastAPI defaults.
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, handle_http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, handle_exception)  # type: ignore[arg-type]
