"""
Arabic NLP API — main application entry point.

Start locally:
    uvicorn main:app --reload --port 8000

Production (Railway):
    Handled via Procfile → gunicorn with uvicorn workers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.api.v1.router import api_router
from app.api.v1.endpoints.health import router as health_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import register_middleware

# ---- Bootstrap ----
configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


# ---- Rate limiter (slowapi wraps limits-per-IP using in-process storage) ----
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_DEFAULT])


# ---- Lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s v%s [%s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )
    # Pre-warm services by importing them — they build any internal state at
    # module load time, so this ensures cold-start latency is paid here, not
    # on the first real request.
    from app.services.preprocessor import preprocessor  # noqa: F401
    from app.services.sentiment import sentiment_analyser  # noqa: F401
    from app.services.dialect_detector import dialect_detector  # noqa: F401
    from app.services.ner import ner_service  # noqa: F401

    logger.info("All NLP services loaded and ready.")
    yield
    logger.info("Shutting down %s.", settings.APP_NAME)


# ---- App factory ----
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "Ahmed Abogabl",
            "url": "https://github.com/AhmedMGabl",
        },
        license_info={
            "name": "MIT",
        },
        servers=[
            {"url": "/", "description": "Current server"},
        ],
        openapi_tags=[
            {
                "name": "Sentiment Analysis",
                "description": "Classify Arabic text as positive, negative, or neutral.",
            },
            {
                "name": "Dialect Detection",
                "description": "Identify the Arabic dialect variant (MSA, Egyptian, Gulf, Levantine, Maghrebi).",
            },
            {
                "name": "Text Preprocessing",
                "description": "Normalise, clean, and tokenise Arabic text.",
            },
            {
                "name": "Named Entity Recognition",
                "description": "Extract persons, locations, organisations, dates, and numbers.",
            },
            {
                "name": "Health",
                "description": "Service health and version information.",
            },
        ],
    )

    # ---- Rate limiter state & middleware ----
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # ---- Custom middleware (CORS, request ID, timing, RapidAPI auth) ----
    register_middleware(app)

    # ---- Exception handlers ----
    register_exception_handlers(app)

    # ---- Routers ----
    # Top-level health — exempt from /v1 prefix so Railway can probe /health
    app.include_router(health_router, prefix="/health", tags=["Health"])

    # All NLP endpoints under /v1
    app.include_router(api_router, prefix="/v1")

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            content={
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "docs": "/docs",
                "health": "/health",
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
