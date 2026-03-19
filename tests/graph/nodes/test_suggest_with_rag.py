"""Tests for suggest node RAG integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.app.graph.nodes.suggest import DestinationList, suggest_destinations
from src.app.schemas import CandidateDestination


def _mock_destination_list() -> DestinationList:
    return DestinationList(destinations=[
        CandidateDestination(
            city="Cancun", iata_code="CUN", country="Mexico",
            reasoning="Playa caribena", climate_match=90, activity_match=80,
        ),
    ])


def _make_llm_mock():
    """Create a mock get_llm that returns a structured LLM."""
    mock_structured = AsyncMock(ainvoke=AsyncMock(return_value=_mock_destination_list()))
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm = MagicMock(return_value=mock_llm)
    return mock_get_llm, mock_structured


class TestSuggestWithRag:
    @patch("src.app.graph.nodes.suggest.query_interactions", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.suggest.query_travel_knowledge", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.suggest.get_llm")
    async def test_suggest_with_rag_context(
        self,
        mock_get_llm,
        mock_query_knowledge,
        mock_query_interactions,
        sample_travel_request,
    ):
        mock_get_llm_fn, mock_structured = _make_llm_mock()
        mock_get_llm.side_effect = mock_get_llm_fn

        # RAG returns knowledge results
        mock_query_knowledge.return_value = [
            {
                "id": "1",
                "score": 0.8,
                "payload": {
                    "text": "Cancun has wonderful beaches and Mayan ruins",
                    "city": "Cancun",
                    "heading": "See",
                },
            },
        ]
        # RAG returns interaction results
        mock_query_interactions.return_value = [
            {
                "id": "int1",
                "score": 0.7,
                "payload": {
                    "text": "Previous trip to Cancun, enjoyed beach activities",
                    "city": "",
                    "heading": "",
                },
            },
        ]

        state = {
            "request": sample_travel_request,
            "suggest_retry_count": 0,
            "user_id": "user-123",
        }
        result = await suggest_destinations(state)

        assert "candidate_destinations" in result
        assert len(result["candidate_destinations"]) == 1

        # Verify RAG-enhanced templates were used (check system message)
        call_messages = mock_structured.ainvoke.call_args[0][0]
        system_content = call_messages[0]["content"]
        user_content = call_messages[1]["content"]
        assert "USA la informaci\u00f3n de contexto" in system_content
        assert "Destination Knowledge" in user_content
        assert "User History" in user_content

    @patch("src.app.graph.nodes.suggest.query_interactions", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.suggest.query_travel_knowledge", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.suggest.get_llm")
    async def test_suggest_falls_back_without_rag(
        self,
        mock_get_llm,
        mock_query_knowledge,
        mock_query_interactions,
        sample_travel_request,
    ):
        mock_get_llm_fn, mock_structured = _make_llm_mock()
        mock_get_llm.side_effect = mock_get_llm_fn

        # RAG returns empty results
        mock_query_knowledge.return_value = []
        mock_query_interactions.return_value = []

        state = {
            "request": sample_travel_request,
            "suggest_retry_count": 0,
            "user_id": "",
        }
        result = await suggest_destinations(state)

        assert "candidate_destinations" in result

        # Verify original (non-RAG) templates were used
        call_messages = mock_structured.ainvoke.call_args[0][0]
        system_content = call_messages[0]["content"]
        # Original template uses accent characters; RAG version does not
        assert "codigo IATA de aeropuerto valido" not in system_content
        # The original template should NOT contain RAG-specific instructions
        assert "USA la informacion de contexto" not in system_content

    @patch("src.app.graph.nodes.suggest.query_interactions", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.suggest.query_travel_knowledge", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.suggest.get_llm")
    async def test_suggest_rag_failure_graceful(
        self,
        mock_get_llm,
        mock_query_knowledge,
        mock_query_interactions,
        sample_travel_request,
    ):
        mock_get_llm_fn, mock_structured = _make_llm_mock()
        mock_get_llm.side_effect = mock_get_llm_fn

        # RAG calls raise an exception
        mock_query_knowledge.side_effect = RuntimeError("Qdrant connection failed")

        state = {
            "request": sample_travel_request,
            "suggest_retry_count": 0,
            "user_id": "user-456",
        }
        result = await suggest_destinations(state)

        # Should still return destinations (graceful degradation)
        assert "candidate_destinations" in result
        assert len(result["candidate_destinations"]) == 1

        # Verify original templates were used (since RAG failed)
        call_messages = mock_structured.ainvoke.call_args[0][0]
        system_content = call_messages[0]["content"]
        assert "USA la informacion de contexto" not in system_content
