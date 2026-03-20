"""POST /v1/sentiment endpoint."""

from typing import Annotated

from fastapi import APIRouter, Body, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.models.requests import SentimentRequest
from app.models.responses import OPENAPI_EXAMPLES, SentimentResponse
from app.services.sentiment import sentiment_analyser

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
settings = get_settings()


@router.post(
    "",
    response_model=SentimentResponse,
    summary="Arabic Sentiment Analysis",
    description=(
        "Analyse the sentiment of an Arabic text and return a label "
        "(positive / negative / neutral) with a confidence score and "
        "the top sentiment-bearing keywords found in the text."
    ),
    responses={
        200: {"description": "Successful sentiment analysis"},
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
                        if k.startswith("sentiment")
                    }
                }
            }
        }
    },
)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def analyse_sentiment(
    request: Request,
    payload: Annotated[SentimentRequest, Body()],
) -> SentimentResponse:
    result = sentiment_analyser.analyse(payload.text)
    return SentimentResponse(**result)
