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

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"

    # Amadeus
    AMADEUS_CLIENT_ID: str = ""
    AMADEUS_CLIENT_SECRET: str = ""
    AMADEUS_ENV: str = "test"  # "test" or "production"

    # App
    REPORT_OUTPUT_DIR: str = "reports"

    @model_validator(mode="after")
    def _warn_empty_credentials(self):
        if not self.AMADEUS_CLIENT_ID or not self.AMADEUS_CLIENT_SECRET:
            logger.warning(
                "Amadeus credentials are empty — flight search will not work. "
                "Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET in .env"
            )
        return self


# Singleton – import this everywhere
settings = Settings()  # type: ignore[call-arg]
