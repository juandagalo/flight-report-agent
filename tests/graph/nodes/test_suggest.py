"""Tests for the suggest node — mocks ChatOpenAI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.app.schemas import CandidateDestination
from src.app.graph.nodes.suggest import DestinationList, suggest_destinations


def _mock_destination_list() -> DestinationList:
    return DestinationList(destinations=[
        CandidateDestination(
            city="Cancún", iata_code="CUN", country="México",
            reasoning="Playa caribeña", climate_match=90, activity_match=80,
        ),
        CandidateDestination(
            city="Punta Cana", iata_code="PUJ", country="Rep. Dominicana",
            reasoning="Resort todo incluido", climate_match=88, activity_match=75,
        ),
    ])


class TestSuggestDestinations:
    async def test_no_request_returns_error(self):
        result = await suggest_destinations({})
        assert "errors" in result
        assert any("No request" in e for e in result["errors"])

    @patch("src.app.graph.nodes.suggest.ChatOpenAI")
    async def test_successful_suggestion(self, mock_chat_cls, sample_travel_request):
        mock_structured = AsyncMock(ainvoke=AsyncMock(return_value=_mock_destination_list()))
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm

        state = {"request": sample_travel_request, "suggest_retry_count": 0}
        result = await suggest_destinations(state)

        assert "candidate_destinations" in result
        assert len(result["candidate_destinations"]) == 2
        assert result["candidate_destinations"][0].iata_code == "CUN"

    @patch("src.app.graph.nodes.suggest.ChatOpenAI")
    async def test_retry_uses_retry_prompt(self, mock_chat_cls, sample_travel_request, sample_candidate):
        mock_structured = AsyncMock(ainvoke=AsyncMock(return_value=_mock_destination_list()))
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm

        state = {
            "request": sample_travel_request,
            "suggest_retry_count": 1,
            "candidate_destinations": [sample_candidate],
        }
        result = await suggest_destinations(state)

        # Verify ainvoke was called (the retry branch formats a different prompt)
        mock_structured.ainvoke.assert_called_once()
        call_messages = mock_structured.ainvoke.call_args[0][0]
        user_content = call_messages[1]["content"]
        assert "ALTERNATIVOS" in user_content or "anteriores" in user_content

    @patch("src.app.graph.nodes.suggest.ChatOpenAI")
    async def test_llm_exception_returns_empty_and_error(self, mock_chat_cls, sample_travel_request):
        mock_structured = AsyncMock(ainvoke=AsyncMock(side_effect=RuntimeError("timeout")))
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm

        state = {"request": sample_travel_request}
        result = await suggest_destinations(state)

        assert result["candidate_destinations"] == []
        assert any("Error al sugerir" in e for e in result["errors"])
