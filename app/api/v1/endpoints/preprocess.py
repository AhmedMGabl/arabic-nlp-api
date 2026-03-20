"""POST /v1/preprocess endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.models.requests import PreprocessRequest
from app.models.responses import OPENAPI_EXAMPLES, PreprocessResponse
from app.services.preprocessor import preprocessor

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
settings = get_settings()


@router.post(
    "",
    response_model=PreprocessResponse,
    summary="Arabic Text Preprocessing",
    description=(
        "Clean and normalise Arabic text. Supports: letter normalisation "
        "(alef variants, teh marbuta, yeh variants), diacritic removal, "
        "punctuation stripping, number removal, and whitespace tokenisation. "
        "Each step is individually togglable via request body flags."
    ),
    responses={
        200: {"description": "Successful preprocessing"},
        422: {"description": "Validation error — text too long or empty"},
        429: {"description": "Rate limit exceeded"},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {"preprocess": OPENAPI_EXAMPLES["preprocess"]}
                }
            }
        }
    },
)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def preprocess_text(
    request: Annotated[PreprocessRequest, Body()],
) -> PreprocessResponse:
    result = preprocessor.process(
        request.text,
        normalize=request.normalize,
        remove_diacritics=request.remove_diacritics,
        remove_punctuation=request.remove_punctuation,
        remove_numbers=request.remove_numbers,
        tokenize=request.tokenize,
    )
    return PreprocessResponse(**result)
