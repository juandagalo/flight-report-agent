"""Tests for the intake node — mocks ChatOpenAI."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from src.app.schemas import DateRange, IntakeResult, TravelRequest
from src.app.graph.nodes.intake import intake_node


class TestIntakeNode:
    async def test_empty_message_passthrough(self):
        """No user_message → no-op, returns empty dict."""
        result = await intake_node({"user_message": ""})
        assert result == {}

    async def test_no_user_message_key_passthrough(self):
        result = await intake_node({})
        assert result == {}

    @patch("src.app.graph.nodes.intake.ChatOpenAI")
    async def test_successful_extraction(self, mock_chat_cls):
        intake_result = IntakeResult(
            preferred_climate="tropical",
            region="Caribe",
            available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
            max_budget=1500.0,
            origin="BOG",
            preferred_activities=["playa", "cultura"],
            num_people=2,
            assumptions=["Se asumió USD"],
        )

        mock_structured = AsyncMock(ainvoke=AsyncMock(return_value=intake_result))
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm

        state = {"user_message": "Quiero ir a la playa en julio desde Bogotá"}
        result = await intake_node(state)

        assert "request" in result
        assert isinstance(result["request"], TravelRequest)
        assert not isinstance(result["request"], IntakeResult)
        assert result["intake_assumptions"] == ["Se asumió USD"]

    @patch("src.app.graph.nodes.intake.ChatOpenAI")
    async def test_llm_exception_returns_error(self, mock_chat_cls):
        mock_structured = AsyncMock(ainvoke=AsyncMock(side_effect=RuntimeError("LLM down")))
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm

        state = {"user_message": "Quiero viajar"}
        result = await intake_node(state)

        assert "errors" in result
        assert any("Error al interpretar" in e for e in result["errors"])
