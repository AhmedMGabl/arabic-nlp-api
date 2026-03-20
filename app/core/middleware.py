"""Custom middleware: request ID injection, timing header, RapidAPI auth gate."""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every request and response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Add X-Process-Time header (milliseconds) to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
        return response


class RapidAPIAuthMiddleware(BaseHTTPMiddleware):
    """
    Validate the RapidAPI proxy secret when RAPIDAPI_PROXY_SECRET is set.
    RapidAPI injects `X-RapidAPI-Proxy-Secret` on every authenticated call.
    Skip /health so Railway health checks still work without the header.
    """

    EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/", "/docs", "/redoc", "/openapi.json"})

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        secret = settings.RAPIDAPI_PROXY_SECRET

        if not secret or request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        received = request.headers.get("X-RapidAPI-Proxy-Secret", "")
        if received != secret:
            logger.warning(
                "Blocked request — invalid RapidAPI proxy secret | path=%s | request_id=%s",
                request.url.path,
                getattr(request.state, "request_id", "-"),
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": {"code": "FORBIDDEN", "message": "Invalid API credentials."}},
            )

        return await call_next(request)


def register_middleware(app: FastAPI) -> None:
    settings = get_settings()

    # Order matters: outermost middleware runs first on the way in,
    # last on the way out.

    # 1. CORS — must be outermost so OPTIONS preflight is handled
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

    # 2. Request ID
    app.add_middleware(RequestIDMiddleware)

    # 3. Timing
    app.add_middleware(TimingMiddleware)

    # 4. RapidAPI auth gate
    app.add_middleware(RapidAPIAuthMiddleware)
