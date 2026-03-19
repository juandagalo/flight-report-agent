"""Tests for enrich node RAG integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.app.graph.nodes.enrich import enrich_data
from src.app.schemas import (
    CandidateDestination,
    DestinationReport,
    FlightOffer,
    WeatherInfo,
)


def _make_report() -> DestinationReport:
    return DestinationReport(
        destination=CandidateDestination(
            city="Cancun",
            iata_code="CUN",
            country="Mexico",
            climate_match=85,
            activity_match=70,
        ),
        flights=[FlightOffer(price=800.0, stops=0)],
    )


class TestEnrichWithRag:
    @patch("src.app.graph.nodes.enrich.query_travel_knowledge", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.enrich.get_llm")
    @patch("src.app.graph.nodes.enrich.get_weather")
    async def test_enrich_with_rag_context(
        self,
        mock_weather,
        mock_get_llm,
        mock_query_knowledge,
        sample_travel_request,
    ):
        weather = WeatherInfo(avg_temp_c=28.0, description="Calido y tropical")
        mock_weather.return_value = weather

        mock_resp = MagicMock(content="- Actividad con RAG\n- Otra actividad")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        mock_get_llm.return_value = mock_llm

        # RAG returns destination-specific knowledge
        mock_query_knowledge.return_value = [
            {
                "id": "1",
                "score": 0.85,
                "payload": {
                    "text": "Cancun has the famous underwater museum MUSA",
                    "city": "Cancun",
                    "heading": "See",
                },
            },
        ]

        report = _make_report()
        state = {
            "request": sample_travel_request,
            "destination_reports": [report],
        }
        result = await enrich_data(state)

        assert result["enriched"] is True
        enriched = result["destination_reports"]
        assert len(enriched) == 1
        assert "Actividad con RAG" in enriched[0].activities_description

        # Verify RAG-enhanced template was used
        call_args = mock_llm.ainvoke.call_args[0][0]
        user_content = call_args[1]["content"]
        assert "Destination Info" in user_content
        assert "informaci\u00f3n de contexto" in user_content

    @patch("src.app.graph.nodes.enrich.query_travel_knowledge", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.enrich.get_llm")
    @patch("src.app.graph.nodes.enrich.get_weather")
    async def test_enrich_falls_back_without_rag(
        self,
        mock_weather,
        mock_get_llm,
        mock_query_knowledge,
        sample_travel_request,
    ):
        weather = WeatherInfo(avg_temp_c=25.0)
        mock_weather.return_value = weather

        mock_resp = MagicMock(content="- Actividad sin RAG")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        mock_get_llm.return_value = mock_llm

        # RAG returns empty results
        mock_query_knowledge.return_value = []

        report = _make_report()
        state = {
            "request": sample_travel_request,
            "destination_reports": [report],
        }
        result = await enrich_data(state)

        assert result["enriched"] is True

        # Verify original (non-RAG) template was used
        call_args = mock_llm.ainvoke.call_args[0][0]
        user_content = call_args[1]["content"]
        assert "Destination Info" not in user_content
        assert "informacion de contexto" not in user_content

    @patch("src.app.graph.nodes.enrich.query_travel_knowledge", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.enrich.get_llm")
    @patch("src.app.graph.nodes.enrich.get_weather")
    async def test_enrich_rag_failure_graceful(
        self,
        mock_weather,
        mock_get_llm,
        mock_query_knowledge,
        sample_travel_request,
    ):
        weather = WeatherInfo(avg_temp_c=28.0)
        mock_weather.return_value = weather

        mock_resp = MagicMock(content="- Actividad normal")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        mock_get_llm.return_value = mock_llm

        # RAG raises an exception
        mock_query_knowledge.side_effect = RuntimeError("Qdrant unavailable")

        report = _make_report()
        state = {
            "request": sample_travel_request,
            "destination_reports": [report],
        }
        result = await enrich_data(state)

        # Should still succeed with fallback template
        assert result["enriched"] is True
        enriched = result["destination_reports"]
        assert len(enriched) == 1
        assert "Actividad normal" in enriched[0].activities_description

        # Verify original template was used (no RAG context)
        call_args = mock_llm.ainvoke.call_args[0][0]
        user_content = call_args[1]["content"]
        assert "Destination Info" not in user_content
