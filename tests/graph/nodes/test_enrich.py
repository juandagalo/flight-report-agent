"""Tests for the enrich node — _compute_score is pure; node mocks weather + LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.app.schemas import (
    CandidateDestination,
    DestinationReport,
    FlightOffer,
    WeatherInfo,
)
from src.app.graph.nodes.enrich import _compute_score, enrich_data


def _report(
    climate_match: int = 80,
    activity_match: int = 70,
    price: float = 500.0,
    stops: int = 0,
    budget: float = 1500.0,
) -> DestinationReport:
    return DestinationReport(
        destination=CandidateDestination(
            city="Test", iata_code="TST", country="X",
            climate_match=climate_match, activity_match=activity_match,
        ),
        flights=[FlightOffer(price=price, stops=stops)] if price > 0 else [],
    )


class TestComputeScore:
    def test_balanced_score(self):
        report = _report(climate_match=80, activity_match=70, price=500, stops=0)
        score = _compute_score(report, budget=1500)
        # climate=80*0.3=24, activity=70*0.3=21, price=int((1-500/1500)*100)*0.25≈16, stops=100*0.15=15
        assert 50 <= score <= 90

    def test_no_flights_scores_lower(self):
        report = _report(price=0)
        report.flights = []
        score = _compute_score(report, budget=1500)
        # price_score=0, stops_score=0
        assert score < 50

    def test_expensive_flight_lowers_score(self):
        cheap = _compute_score(_report(price=200), budget=1500)
        expensive = _compute_score(_report(price=1400), budget=1500)
        assert cheap > expensive

    def test_free_flight_max_price_score(self):
        report = _report(price=0.01)
        score = _compute_score(report, budget=1500)
        # price_ratio ≈ 1.0 → price_score ≈ 100
        assert score > 60

    def test_stops_penalty(self):
        nonstop = _compute_score(_report(stops=0), budget=1500)
        with_stops = _compute_score(_report(stops=2), budget=1500)
        assert nonstop > with_stops


class TestEnrichData:
    async def test_no_request(self):
        result = await enrich_data({})
        assert "errors" in result

    async def test_no_reports_returns_enriched(self, sample_travel_request):
        state = {"request": sample_travel_request, "destination_reports": []}
        result = await enrich_data(state)
        assert result["enriched"] is True

    @patch("src.app.graph.nodes.enrich.ChatOpenAI")
    @patch("src.app.graph.nodes.enrich.get_weather")
    async def test_attaches_weather_and_activities(
        self, mock_weather, mock_chat_cls, sample_travel_request, sample_candidate, sample_flight_offer
    ):
        weather = WeatherInfo(avg_temp_c=28.0, description="Cálido y tropical, seco")
        mock_weather.return_value = weather

        mock_resp = MagicMock(content="• Actividad 1\n• Actividad 2")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        mock_chat_cls.return_value = mock_llm

        report = DestinationReport(destination=sample_candidate, flights=[sample_flight_offer])
        state = {
            "request": sample_travel_request,
            "destination_reports": [report],
        }
        result = await enrich_data(state)

        assert result["enriched"] is True
        enriched = result["destination_reports"]
        assert len(enriched) == 1
        assert enriched[0].weather.avg_temp_c == 28.0
        assert "Actividad 1" in enriched[0].activities_description
        assert enriched[0].overall_score > 0

    @patch("src.app.graph.nodes.enrich.ChatOpenAI")
    @patch("src.app.graph.nodes.enrich.get_weather")
    async def test_llm_failure_fallback(
        self, mock_weather, mock_chat_cls, sample_travel_request, sample_candidate, sample_flight_offer
    ):
        mock_weather.return_value = WeatherInfo()

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_chat_cls.return_value = mock_llm

        report = DestinationReport(destination=sample_candidate, flights=[sample_flight_offer])
        state = {
            "request": sample_travel_request,
            "destination_reports": [report],
        }
        result = await enrich_data(state)

        assert result["enriched"] is True
        assert "no disponible" in result["destination_reports"][0].activities_description
