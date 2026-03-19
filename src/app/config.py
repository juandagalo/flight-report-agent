"""Application configuration loaded from environment variables."""

import logging

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Global settings – values come from .env or real env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider
    LLM_PROVIDER: str = "openai"  # "openai" or "claude"

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Amadeus
    AMADEUS_CLIENT_ID: str = ""
    AMADEUS_CLIENT_SECRET: str = ""
    AMADEUS_ENV: str = "test"  # "test" or "production"

    # Qdrant
    QDRANT_PATH: str = "data/qdrant"
    QDRANT_COLLECTION_KNOWLEDGE: str = "travel_knowledge"
    QDRANT_COLLECTION_INTERACTIONS: str = "interactions"
    EMBEDDING_DIMENSION: int = 1536

    # App
    REPORT_OUTPUT_DIR: str = "reports"

    @model_validator(mode="after")
    def _warn_empty_credentials(self):
        if not self.AMADEUS_CLIENT_ID or not self.AMADEUS_CLIENT_SECRET:
            logger.warning(
                "Amadeus credentials are empty — flight search will not work. "
                "Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET in .env"
            )
        if self.LLM_PROVIDER.lower().strip() == "claude" and not self.ANTHROPIC_API_KEY:
            logger.warning(
                "LLM_PROVIDER is 'claude' but ANTHROPIC_API_KEY is empty — "
                "LLM calls will fail. Set ANTHROPIC_API_KEY in .env"
            )
        return self


# Singleton – import this everywhere
settings = Settings()  # type: ignore[call-arg]
