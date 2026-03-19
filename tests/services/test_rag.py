"""Tests for RAG context retrieval helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.app.services.rag import (
    format_rag_context,
    query_interactions,
    query_travel_knowledge,
)


class TestQueryTravelKnowledge:
    @patch("src.app.services.rag.search")
    @patch("src.app.services.rag.embed_query", new_callable=AsyncMock)
    @patch("src.app.services.rag.ensure_all_collections")
    async def test_calls_embed_and_search(
        self, mock_ensure, mock_embed, mock_search
    ):
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_search.return_value = [
            {"id": "1", "score": 0.8, "payload": {"text": "Beach info"}}
        ]

        result = await query_travel_knowledge("tropical beach activities")

        mock_ensure.assert_called_once()
        mock_embed.assert_called_once_with("tropical beach activities")
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert call_kwargs.kwargs["limit"] == 5
        assert call_kwargs.kwargs["score_threshold"] == 0.3
        assert call_kwargs.kwargs["filter_conditions"] is None
        assert len(result) == 1

    @patch("src.app.services.rag.search")
    @patch("src.app.services.rag.embed_query", new_callable=AsyncMock)
    @patch("src.app.services.rag.ensure_all_collections")
    async def test_with_iata_filter(self, mock_ensure, mock_embed, mock_search):
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = []

        await query_travel_knowledge("beach", destination_iata="CUN")

        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["filter_conditions"] == {"iata": "CUN"}

    @patch("src.app.services.rag.search")
    @patch("src.app.services.rag.embed_query", new_callable=AsyncMock)
    @patch("src.app.services.rag.ensure_all_collections")
    async def test_custom_limit(self, mock_ensure, mock_embed, mock_search):
        mock_embed.return_value = [0.1]
        mock_search.return_value = []

        await query_travel_knowledge("beach", limit=10)

        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["limit"] == 10


class TestQueryInteractions:
    async def test_empty_user_id_returns_empty(self):
        result = await query_interactions("beach trip", user_id="")
        assert result == []

    @patch("src.app.services.rag.search")
    @patch("src.app.services.rag.embed_query", new_callable=AsyncMock)
    @patch("src.app.services.rag.ensure_all_collections")
    async def test_calls_search_with_user_filter(
        self, mock_ensure, mock_embed, mock_search
    ):
        mock_embed.return_value = [0.5, 0.6]
        mock_search.return_value = [
            {"id": "int1", "score": 0.7, "payload": {"text": "Previous trip"}}
        ]

        result = await query_interactions("beach", user_id="user-42", limit=3)

        mock_ensure.assert_called_once()
        mock_embed.assert_called_once_with("beach")
        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["filter_conditions"] == {"user_id": "user-42"}
        assert call_kwargs.kwargs["limit"] == 3
        assert len(result) == 1


class TestFormatRagContext:
    def test_empty_results_returns_empty_string(self):
        assert format_rag_context([]) == ""

    def test_formats_results_with_label(self):
        results = [
            {
                "id": "1",
                "score": 0.9,
                "payload": {
                    "text": "Cancun has beautiful beaches",
                    "city": "Cancun",
                    "heading": "See",
                },
            },
            {
                "id": "2",
                "score": 0.8,
                "payload": {
                    "text": "Great diving spots",
                    "city": "Cancun",
                    "heading": "Do",
                },
            },
        ]
        formatted = format_rag_context(results, label="Destination Knowledge")

        assert "--- Destination Knowledge ---" in formatted
        assert "--- End Destination Knowledge ---" in formatted
        assert "Cancun has beautiful beaches" in formatted
        assert "[Cancun - See]" in formatted
        assert "Great diving spots" in formatted
        assert "[Cancun - Do]" in formatted

    def test_truncates_long_text(self):
        long_text = "A" * 600
        results = [
            {
                "id": "1",
                "score": 0.9,
                "payload": {
                    "text": long_text,
                    "city": "TestCity",
                    "heading": "Info",
                },
            },
        ]
        formatted = format_rag_context(results, label="Context")

        # The text should be truncated at 500 characters
        # Find the line with the text content
        lines = formatted.split("\n")
        text_line = [l for l in lines if "AAA" in l][0]
        # The line will have "- " prefix + 500 chars of A + " [TestCity - Info]"
        assert len(long_text[:500]) == 500
        assert "A" * 501 not in text_line

    def test_handles_missing_payload_fields(self):
        results = [
            {
                "id": "1",
                "score": 0.9,
                "payload": {"text": "Some text"},
            },
        ]
        formatted = format_rag_context(results, label="Test")

        assert "Some text" in formatted
        assert "--- Test ---" in formatted
        assert "--- End Test ---" in formatted

    def test_default_label(self):
        results = [
            {
                "id": "1",
                "score": 0.5,
                "payload": {"text": "data"},
            },
        ]
        formatted = format_rag_context(results)

        assert "--- Context ---" in formatted
        assert "--- End Context ---" in formatted
