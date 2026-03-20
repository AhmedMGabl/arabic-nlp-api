"""POST /v1/preprocess endpoint."""

import time
from typing import Annotated

from fastapi import APIRouter, Body, Request
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
    request: Request,
    payload: Annotated[PreprocessRequest, Body()],
) -> PreprocessResponse:
    t0 = time.perf_counter()
    result = preprocessor.process(
        payload.text,
        normalize=payload.normalize,
        remove_diacritics=payload.remove_diacritics,
        remove_punctuation=payload.remove_punctuation,
        remove_numbers=payload.remove_numbers,
        tokenize=payload.tokenize,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    result["meta"] = {
        "char_count": len(payload.text),
        "processing_time_ms": elapsed_ms,
    }
    return PreprocessResponse(**result)
