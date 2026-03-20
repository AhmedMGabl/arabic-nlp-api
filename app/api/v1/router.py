"""V1 API router — wires all endpoint sub-routers under /v1."""

from fastapi import APIRouter

from app.api.v1.endpoints import dialect, entities, health, preprocess, sentiment

api_router = APIRouter()

api_router.include_router(
    sentiment.router,
    prefix="/sentiment",
    tags=["Sentiment Analysis"],
)

api_router.include_router(
    dialect.router,
    prefix="/detect-dialect",
    tags=["Dialect Detection"],
)

api_router.include_router(
    preprocess.router,
    prefix="/preprocess",
    tags=["Text Preprocessing"],
)

api_router.include_router(
    entities.router,
    prefix="/entities",
    tags=["Named Entity Recognition"],
)

# Health check lives at /health (top-level), included here so it shows
# in the v1 OpenAPI docs but also registered at root level in main.py.
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)
