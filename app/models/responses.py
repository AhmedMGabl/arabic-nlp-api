"""Response models for all NLP endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class MetaBlock(BaseModel):
    """Common metadata attached to every successful response."""

    char_count: int = Field(..., description="Character count of the input text.")
    processing_time_ms: float = Field(..., description="Server processing time in milliseconds.")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    environment: str


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

SentimentLabel = Literal["positive", "negative", "neutral"]


class SentimentResponse(BaseModel):
    sentiment: SentimentLabel = Field(..., description="Overall sentiment label.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score between 0 and 1.",
    )
    positive_score: float = Field(..., ge=0.0, le=1.0)
    negative_score: float = Field(..., ge=0.0, le=1.0)
    neutral_score: float = Field(..., ge=0.0, le=1.0)
    key_words: list[str] = Field(
        default_factory=list,
        description="Top sentiment-bearing words found in the text.",
    )
    meta: MetaBlock


# ---------------------------------------------------------------------------
# Dialect detection
# ---------------------------------------------------------------------------

DialectCode = Literal["MSA", "EGY", "GULF", "LEV", "MAG"]


class DialectScore(BaseModel):
    code: DialectCode
    name_en: str
    name_ar: str
    score: float = Field(..., ge=0.0, le=1.0, description="Probability estimate.")


class DialectResponse(BaseModel):
    dialect: DialectCode = Field(..., description="Top predicted dialect code.")
    dialect_name_en: str
    dialect_name_ar: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: list[DialectScore] = Field(..., description="Scores for all five dialects.")
    meta: MetaBlock


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class PreprocessResponse(BaseModel):
    original: str
    processed: str
    tokens: list[str] = Field(default_factory=list)
    token_count: int
    operations_applied: list[str] = Field(
        default_factory=list,
        description="List of preprocessing steps that were applied.",
    )
    meta: MetaBlock


# ---------------------------------------------------------------------------
# Named Entity Recognition
# ---------------------------------------------------------------------------

EntityType = Literal["PERSON", "LOCATION", "ORGANIZATION", "DATE", "NUMBER", "MISC"]


class Entity(BaseModel):
    text: str = Field(..., description="The entity surface form as found in the text.")
    entity_type: EntityType
    start: int = Field(..., description="Character start index in the original text.")
    end: int = Field(..., description="Character end index (exclusive).")
    confidence: float = Field(..., ge=0.0, le=1.0)


class NERResponse(BaseModel):
    entities: list[Entity]
    entity_count: int
    entity_types_found: list[str] = Field(
        default_factory=list,
        description="Unique entity types found.",
    )
    meta: MetaBlock


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# OpenAPI example responses helper
# ---------------------------------------------------------------------------

OPENAPI_EXAMPLES: dict[str, Any] = {
    "sentiment_positive": {
        "summary": "Positive review",
        "value": {"text": "هذا المنتج رائع جداً وأنصح به بشدة، جودة ممتازة!"},
    },
    "sentiment_negative": {
        "summary": "Negative complaint",
        "value": {"text": "الخدمة سيئة جداً وكانت التجربة فظيعة"},
    },
    "dialect_egy": {
        "summary": "Egyptian dialect",
        "value": {"text": "عاملين إيه يا جماعة؟ الأكل كان تمام أوي"},
    },
    "dialect_gulf": {
        "summary": "Gulf dialect",
        "value": {"text": "شلونك؟ الحين وايد تعبان من الشغل"},
    },
    "dialect_lev": {
        "summary": "Levantine dialect",
        "value": {"text": "شو بدك؟ أنا هلق رح روح عالبيت"},
    },
    "preprocess": {
        "summary": "Text with diacritics and punctuation",
        "value": {
            "text": "الرَّجُلُ يَقْرَأُ الكِتَابَ، وَيَفْهَمُ مَا فِيهِ!",
            "normalize": True,
            "remove_diacritics": True,
            "remove_punctuation": True,
            "tokenize": True,
        },
    },
    "ner": {
        "summary": "News snippet",
        "value": {
            "text": "أعلن الدكتور محمد العبدالله في الرياض عن إطلاق شركة التقنية الجديدة يوم الاثنين"
        },
    },
}
