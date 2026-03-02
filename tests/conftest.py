"""Shared fixtures and environment setup for the test suite."""

from __future__ import annotations

import os

# Set required env vars BEFORE any app module is imported.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("AMADEUS_CLIENT_ID", "test-id")
os.environ.setdefault("AMADEUS_CLIENT_SECRET", "test-secret")

from datetime import date

import pytest

from src.app.schemas import (
    CandidateDestination,
    DateRange,
    DestinationReport,
    FlightOffer,
    TravelRequest,
    WeatherInfo,
)


@pytest.fixture
def sample_date_range() -> DateRange:
    return DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))


@pytest.fixture
def sample_travel_request(sample_date_range: DateRange) -> TravelRequest:
    return TravelRequest(
        preferred_climate="tropical",
        region="Caribe",
        available_dates=[sample_date_range],
        max_budget=1500.0,
        currency="USD",
        origin="BOG",
        preferred_activities=["playa", "cultura"],
        num_people=2,
    )


@pytest.fixture
def sample_candidate() -> CandidateDestination:
    return CandidateDestination(
        city="Cancún",
        iata_code="CUN",
        country="México",
        reasoning="Destino caribeño popular",
        climate_match=85,
        activity_match=70,
    )


@pytest.fixture
def sample_flight_offer() -> FlightOffer:
    return FlightOffer(
        airline="Avianca",
        price=800.0,
        currency="USD",
        departure="2026-07-01T08:00:00",
        arrival="2026-07-01T12:00:00",
        duration="PT4H0M",
        stops=0,
        return_departure="2026-07-15T14:00:00",
        return_arrival="2026-07-15T18:00:00",
        return_duration="PT4H0M",
        return_stops=0,
    )


@pytest.fixture
def sample_weather() -> WeatherInfo:
    return WeatherInfo(
        avg_temp_c=28.5,
        min_temp_c=24.0,
        max_temp_c=33.0,
        avg_precipitation_mm=3.2,
        description="Cálido y tropical, lluvias ocasionales",
    )


@pytest.fixture
def sample_destination_report(
    sample_candidate: CandidateDestination,
    sample_flight_offer: FlightOffer,
    sample_weather: WeatherInfo,
) -> DestinationReport:
    return DestinationReport(
        destination=sample_candidate,
        flights=[sample_flight_offer],
        weather=sample_weather,
        activities_description="• Visitar ruinas mayas\n• Snorkel en arrecifes",
        overall_score=75,
    )


@pytest.fixture
def sample_travel_state(
    sample_travel_request: TravelRequest,
    sample_candidate: CandidateDestination,
    sample_destination_report: DestinationReport,
) -> dict:
    return {
        "user_message": "",
        "request": sample_travel_request,
        "validated": True,
        "validation_errors": [],
        "candidate_destinations": [sample_candidate],
        "destination_reports": [sample_destination_report],
        "enriched": False,
        "report_path": "",
        "suggest_retry_count": 0,
        "errors": [],
    }
