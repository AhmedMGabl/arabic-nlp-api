"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App metadata ----
    APP_NAME: str = "Arabic NLP API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Production-ready Arabic NLP API providing sentiment analysis, "
        "dialect detection, text preprocessing, and named entity recognition."
    )
    ENVIRONMENT: Literal["development", "staging", "production"] = "production"
    DEBUG: bool = False

    # ---- Server ----
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)

    # ---- CORS ----
    # Comma-separated list of allowed origins. Use "*" for wide-open (RapidAPI proxy).
    ALLOWED_ORIGINS: str = "*"
    ALLOWED_METHODS: str = "GET,POST,OPTIONS"
    ALLOWED_HEADERS: str = "*"

    # ---- Rate limiting (slowapi / in-process) ----
    RATE_LIMIT_DEFAULT: str = "60/minute"      # default per-endpoint limit
    RATE_LIMIT_BURST: str = "10/second"        # burst guard

    # ---- RapidAPI ----
    # When deployed via RapidAPI the platform injects this header on every
    # authenticated request. Set to empty string to disable the check.
    RAPIDAPI_PROXY_SECRET: str = ""

    # ---- Logging ----
    LOG_LEVEL: str = "INFO"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def cors_methods(self) -> list[str]:
        return [m.strip() for m in self.ALLOWED_METHODS.split(",") if m.strip()]

    @property
    def cors_headers(self) -> list[str]:
        return [h.strip() for h in self.ALLOWED_HEADERS.split(",") if h.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
