"""Tests for the LLM provider factory function."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.app.services.llm_provider import get_llm


def _mock_settings(**overrides):
    """Build a mock settings object with sensible defaults."""
    defaults = {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_MODEL": "gpt-4o",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "ANTHROPIC_MODEL": "claude-sonnet-4-20250514",
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestGetLlm:
    @patch("src.app.services.llm_provider.settings", _mock_settings(LLM_PROVIDER="openai"))
    def test_get_llm_openai_returns_instance(self):
        """With LLM_PROVIDER=openai, the factory calls ChatOpenAI."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = get_llm()
            mock_cls.assert_called_once_with(
                model="gpt-4o",
                temperature=0.7,
                api_key="test-openai-key",
            )
            assert result == mock_cls.return_value

    @patch("src.app.services.llm_provider.settings", _mock_settings(LLM_PROVIDER="claude"))
    def test_get_llm_claude(self):
        """With LLM_PROVIDER=claude, returns ChatAnthropic instance."""
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = get_llm()
            mock_cls.assert_called_once_with(
                model="claude-sonnet-4-20250514",
                temperature=0.7,
                api_key="test-anthropic-key",
            )
            assert result == mock_cls.return_value

    @patch("src.app.services.llm_provider.settings", _mock_settings(LLM_PROVIDER="gemini"))
    def test_get_llm_unknown_provider_raises(self):
        """With LLM_PROVIDER=gemini, raises ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER: 'gemini'"):
            get_llm()

    @patch("src.app.services.llm_provider.settings", _mock_settings(LLM_PROVIDER="openai"))
    def test_get_llm_temperature_passed(self):
        """Verify the temperature kwarg is forwarded."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm(temperature=0.3)
            mock_cls.assert_called_once_with(
                model="gpt-4o",
                temperature=0.3,
                api_key="test-openai-key",
            )

    @patch("src.app.services.llm_provider.settings", _mock_settings(LLM_PROVIDER="Claude"))
    def test_get_llm_case_insensitive(self):
        """LLM_PROVIDER=Claude (capital C) works."""
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = get_llm()
            mock_cls.assert_called_once()
            assert result == mock_cls.return_value

    @patch("src.app.services.llm_provider.settings", _mock_settings(LLM_PROVIDER=" openai "))
    def test_get_llm_whitespace_stripped(self):
        """LLM_PROVIDER=' openai ' (with whitespace) works."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = get_llm()
            mock_cls.assert_called_once()
            assert result == mock_cls.return_value
