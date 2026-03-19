"""Tests for the embeddings service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.services.embeddings import (
    embed_query,
    embed_texts,
    get_embeddings,
    reset_embeddings,
)


def _mock_settings(**overrides):
    """Build a mock settings object with sensible defaults."""
    defaults = {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-openai-key",
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestGetEmbeddings:
    def setup_method(self):
        """Reset the embeddings singleton before each test."""
        reset_embeddings()

    def teardown_method(self):
        """Reset after each test to avoid leaking state."""
        reset_embeddings()

    @patch("src.app.services.embeddings.settings", _mock_settings(LLM_PROVIDER="openai"))
    def test_get_embeddings_returns_openai_embeddings(self):
        """Verify returns OpenAIEmbeddings instance with model text-embedding-3-small."""
        with patch("langchain_openai.OpenAIEmbeddings") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            result = get_embeddings()

            mock_cls.assert_called_once_with(
                model="text-embedding-3-small",
                api_key="test-openai-key",
            )
            assert result == mock_instance

    @patch(
        "src.app.services.embeddings.settings",
        _mock_settings(LLM_PROVIDER="claude", OPENAI_API_KEY="test-openai-key"),
    )
    def test_get_embeddings_uses_openai_regardless_of_provider(self):
        """With LLM_PROVIDER=claude, still returns OpenAIEmbeddings."""
        with patch("langchain_openai.OpenAIEmbeddings") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            result = get_embeddings()

            mock_cls.assert_called_once_with(
                model="text-embedding-3-small",
                api_key="test-openai-key",
            )
            assert result == mock_instance


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_embed_texts_calls_aembed_documents(self):
        """Mock get_embeddings() and verify aembed_documents is called with input list."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        with patch("src.app.services.embeddings.get_embeddings", return_value=mock_embeddings):
            result = await embed_texts(["hello", "world"])

        mock_embeddings.aembed_documents.assert_called_once_with(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_embed_texts_returns_vectors(self):
        """Mock returns specific vectors; verify the function returns them unchanged."""
        expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(return_value=expected)

        with patch("src.app.services.embeddings.get_embeddings", return_value=mock_embeddings):
            result = await embed_texts(["text1", "text2"])

        assert result == expected


class TestEmbedQuery:
    @pytest.mark.asyncio
    async def test_embed_query_calls_aembed_query(self):
        """Mock get_embeddings() and verify aembed_query is called."""
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        with patch("src.app.services.embeddings.get_embeddings", return_value=mock_embeddings):
            result = await embed_query("hello")

        mock_embeddings.aembed_query.assert_called_once_with("hello")
        assert result == [0.1, 0.2, 0.3]
