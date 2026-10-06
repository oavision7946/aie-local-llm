import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.services.engines.errors import EngineError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        logger.warning("Validation error on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=422, content={"detail": "Invalid request payload."})

    @app.exception_handler(EngineError)
    async def _engine_error(request: Request, exc: EngineError):
        logger.error("Engine error on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=502, content={"detail": exc.safe_message})

    @app.exception_handler(HTTPException)
    async def _http_exception(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
