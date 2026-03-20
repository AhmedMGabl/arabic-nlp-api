"""Custom exception hierarchy and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class ArabicNLPError(Exception):
    """Base exception for all domain errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ValidationError(ArabicNLPError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "VALIDATION_ERROR"


class TextTooLongError(ValidationError):
    error_code = "TEXT_TOO_LONG"


class EmptyTextError(ValidationError):
    error_code = "EMPTY_TEXT"


class UnsupportedLanguageError(ArabicNLPError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "UNSUPPORTED_LANGUAGE"


def _error_body(exc: ArabicNLPError) -> dict:
    return {
        "error": {
            "code": exc.error_code,
            "message": exc.message,
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ArabicNLPError)
    async def handle_domain_error(
        request: Request, exc: ArabicNLPError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc),
        )

    @app.exception_handler(Exception)
    async def handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred.",
                }
            },
        )
