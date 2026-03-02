"""Tests for the search_flights node — mocks amadeus_client.search_flights."""

from __future__ import annotations

from unittest.mock import patch

from src.app.schemas import CandidateDestination, FlightOffer
from src.app.graph.nodes.search_flights import search_flights_node


def _offer(price: float, stops: int = 0) -> FlightOffer:
    return FlightOffer(price=price, currency="USD", stops=stops, airline="Test Air")


class TestSearchFlightsNode:
    async def test_no_request(self):
        result = await search_flights_node({})
        assert result["destination_reports"] == []
        assert any("No request" in e for e in result["errors"])

    async def test_no_candidates(self, sample_travel_request):
        state = {"request": sample_travel_request, "candidate_destinations": []}
        result = await search_flights_node(state)
        assert result["destination_reports"] == []
        assert any("No hay destinos" in e for e in result["errors"])

    @patch("src.app.graph.nodes.search_flights.search_flights")
    async def test_budget_filtering(self, mock_search, sample_travel_request, sample_candidate):
        """Flights over budget*num_people are excluded."""
        # budget=1500, num_people=2 → threshold=3000
        mock_search.return_value = [_offer(2500), _offer(3500)]

        state = {
            "request": sample_travel_request,
            "candidate_destinations": [sample_candidate],
        }
        result = await search_flights_node(state)

        # Only the 2500 offer is affordable (2500 <= 3000)
        assert len(result["destination_reports"]) == 1
        assert len(result["destination_reports"][0].flights) == 1
        assert result["destination_reports"][0].flights[0].price == 2500

    @patch("src.app.graph.nodes.search_flights.search_flights")
    async def test_destination_excluded_when_all_over_budget(self, mock_search, sample_travel_request, sample_candidate):
        """Destination dropped entirely if all flights exceed budget."""
        mock_search.return_value = [_offer(5000), _offer(6000)]

        state = {
            "request": sample_travel_request,
            "candidate_destinations": [sample_candidate],
        }
        result = await search_flights_node(state)
        assert result["destination_reports"] == []

    @patch("src.app.graph.nodes.search_flights.search_flights")
    async def test_multiple_destinations(self, mock_search, sample_travel_request):
        cand1 = CandidateDestination(city="A", iata_code="AAA", country="X")
        cand2 = CandidateDestination(city="B", iata_code="BBB", country="Y")

        mock_search.side_effect = [
            [_offer(1000)],  # affordable for A
            [_offer(1000)],  # affordable for B
        ]

        state = {
            "request": sample_travel_request,
            "candidate_destinations": [cand1, cand2],
        }
        result = await search_flights_node(state)
        assert len(result["destination_reports"]) == 2
