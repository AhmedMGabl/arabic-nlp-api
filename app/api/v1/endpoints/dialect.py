"""POST /v1/detect-dialect endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.models.requests import DialectRequest
from app.models.responses import OPENAPI_EXAMPLES, DialectResponse
from app.services.dialect_detector import dialect_detector

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
settings = get_settings()


@router.post(
    "",
    response_model=DialectResponse,
    summary="Arabic Dialect Detection",
    description=(
        "Detect the Arabic dialect of a given text. Returns a top dialect label "
        "with confidence and a probability distribution across all five dialects: "
        "MSA (Modern Standard Arabic), EGY (Egyptian), GULF, LEV (Levantine), "
        "and MAG (Maghrebi)."
    ),
    responses={
        200: {"description": "Successful dialect detection"},
        422: {"description": "Validation error — text too long or empty"},
        429: {"description": "Rate limit exceeded"},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        k: v
                        for k, v in OPENAPI_EXAMPLES.items()
                        if k.startswith("dialect")
                    }
                }
            }
        }
    },
)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def detect_dialect(
    request: Annotated[DialectRequest, Body()],
) -> DialectResponse:
    result = dialect_detector.detect(request.text)
    return DialectResponse(**result)
