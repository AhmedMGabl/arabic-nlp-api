"""Request models for all NLP endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Maximum text length accepted by any endpoint (Railway free-tier friendly)
MAX_TEXT_LENGTH = 5_000


class TextRequest(BaseModel):
    """Shared base for all single-text endpoints."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Arabic text to analyse. Max 5,000 characters.",
        examples=["هذا المنتج رائع جداً وأنصح به بشدة"],
    )

    @field_validator("text")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty or whitespace only")
        return v


class SentimentRequest(TextRequest):
    """Request body for POST /sentiment."""


class DialectRequest(TextRequest):
    """Request body for POST /detect-dialect."""


class PreprocessRequest(TextRequest):
    """Request body for POST /preprocess."""

    normalize: bool = Field(
        default=True,
        description="Normalize Arabic letters (e.g. أإآ -> ا, ة -> ه).",
    )
    remove_diacritics: bool = Field(
        default=True,
        description="Remove tashkeel (harakat) diacritic marks.",
    )
    remove_punctuation: bool = Field(
        default=True,
        description="Remove punctuation characters.",
    )
    remove_numbers: bool = Field(
        default=False,
        description="Remove Arabic-Indic and Western digit sequences.",
    )
    tokenize: bool = Field(
        default=True,
        description="Return a list of whitespace-split tokens.",
    )


class NERRequest(TextRequest):
    """Request body for POST /entities."""
