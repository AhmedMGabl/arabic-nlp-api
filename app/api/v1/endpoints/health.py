"""GET /health endpoint — used by Railway and RapidAPI uptime checks."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.responses import HealthResponse

router = APIRouter()
settings = get_settings()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description=(
        "Returns the current service health status, API version, and environment. "
        "This endpoint is exempt from authentication and rate limiting."
    ),
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )
