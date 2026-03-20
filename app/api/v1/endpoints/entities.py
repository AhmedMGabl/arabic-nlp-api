"""POST /v1/entities endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.models.requests import NERRequest
from app.models.responses import OPENAPI_EXAMPLES, NERResponse
from app.services.ner import ner_service

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
settings = get_settings()


@router.post(
    "",
    response_model=NERResponse,
    summary="Arabic Named Entity Recognition",
    description=(
        "Extract named entities from Arabic text. Recognised entity types: "
        "PERSON, LOCATION, ORGANIZATION, DATE, NUMBER. "
        "Returns each entity with its character span and a confidence score."
    ),
    responses={
        200: {"description": "Successful entity extraction"},
        422: {"description": "Validation error — text too long or empty"},
        429: {"description": "Rate limit exceeded"},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {"ner": OPENAPI_EXAMPLES["ner"]}
                }
            }
        }
    },
)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def extract_entities(
    request: Annotated[NERRequest, Body()],
) -> NERResponse:
    result = ner_service.extract(request.text)
    return NERResponse(**result)
