import logging

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    QnAException,
    DatabaseException,
    EmbeddingException,
    LLMException,
    VectorStoreException
)


logger = logging.getLogger(__name__)

async def qna_exception_handler(_: Request, exc: QnAException) -> JSONResponse:
    logger.error("QnA Exception: %s", exc.message, extra={"details": exc.details})
    
    status_code = 500
    if isinstance(exc, DatabaseException):
        status_code = 503
    elif isinstance(exc, (EmbeddingException, LLMException, VectorStoreException)):
        status_code = 502
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTP Exception: %s", exc.detail)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "details": {}
        }
    )


async def general_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", str(exc), exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {}
        }
    )
