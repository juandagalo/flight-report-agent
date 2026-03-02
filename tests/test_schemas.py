"""Tests for Pydantic schema models."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.app.schemas import (
    CandidateDestination,
    DateRange,
    DestinationReport,
    FlightOffer,
    IntakeResult,
    TravelRequest,
    WeatherInfo,
)


class TestDateRange:
    def test_valid(self):
        dr = DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))
        assert dr.date_from == date(2026, 7, 1)
        assert dr.date_to == date(2026, 7, 15)


class TestTravelRequest:
    def test_valid_construction(self, sample_travel_request: TravelRequest):
        assert sample_travel_request.origin == "BOG"
        assert sample_travel_request.num_people == 2
        assert sample_travel_request.max_budget == 1500.0

    def test_origin_min_length(self):
        with pytest.raises(ValidationError, match="origin"):
            TravelRequest(
                preferred_climate="tropical",
                region="Caribe",
                available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
                max_budget=1000,
                origin="BO",
                preferred_activities=["playa"],
                num_people=1,
            )

    def test_origin_max_length(self):
        with pytest.raises(ValidationError, match="origin"):
            TravelRequest(
                preferred_climate="tropical",
                region="Caribe",
                available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
                max_budget=1000,
                origin="BOGX",
                preferred_activities=["playa"],
                num_people=1,
            )

    def test_budget_must_be_positive(self):
        with pytest.raises(ValidationError, match="max_budget"):
            TravelRequest(
                preferred_climate="tropical",
                region="Caribe",
                available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
                max_budget=0,
                origin="BOG",
                preferred_activities=["playa"],
                num_people=1,
            )

    def test_num_people_lower_bound(self):
        with pytest.raises(ValidationError, match="num_people"):
            TravelRequest(
                preferred_climate="tropical",
                region="Caribe",
                available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
                max_budget=1000,
                origin="BOG",
                preferred_activities=["playa"],
                num_people=0,
            )

    def test_num_people_upper_bound(self):
        with pytest.raises(ValidationError, match="num_people"):
            TravelRequest(
                preferred_climate="tropical",
                region="Caribe",
                available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
                max_budget=1000,
                origin="BOG",
                preferred_activities=["playa"],
                num_people=10,
            )

    def test_empty_dates_rejected(self):
        with pytest.raises(ValidationError, match="available_dates"):
            TravelRequest(
                preferred_climate="tropical",
                region="Caribe",
                available_dates=[],
                max_budget=1000,
                origin="BOG",
                preferred_activities=["playa"],
                num_people=1,
            )

    def test_empty_activities_rejected(self):
        with pytest.raises(ValidationError, match="preferred_activities"):
            TravelRequest(
                preferred_climate="tropical",
                region="Caribe",
                available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
                max_budget=1000,
                origin="BOG",
                preferred_activities=[],
                num_people=1,
            )


class TestCandidateDestination:
    def test_climate_match_rejects_over_100(self):
        with pytest.raises(ValidationError, match="climate_match"):
            CandidateDestination(
                city="Test", iata_code="TST", country="X",
                climate_match=101,
            )

    def test_activity_match_rejects_negative(self):
        with pytest.raises(ValidationError, match="activity_match"):
            CandidateDestination(
                city="Test", iata_code="TST", country="X",
                activity_match=-1,
            )


class TestIntakeResult:
    def test_inherits_travel_request(self):
        result = IntakeResult(
            preferred_climate="tropical",
            region="Caribe",
            available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
            max_budget=1000,
            origin="BOG",
            preferred_activities=["playa"],
            num_people=1,
            assumptions=["Assumed USD"],
        )
        assert isinstance(result, TravelRequest)
        assert result.assumptions == ["Assumed USD"]

    def test_assumptions_default_empty(self):
        result = IntakeResult(
            preferred_climate="tropical",
            region="Caribe",
            available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
            max_budget=1000,
            origin="BOG",
            preferred_activities=["playa"],
            num_people=1,
        )
        assert result.assumptions == []


class TestFlightOffer:
    def test_defaults(self):
        offer = FlightOffer()
        assert offer.airline == ""
        assert offer.price == 0.0
        assert offer.currency == "USD"
        assert offer.stops == 0
        assert offer.return_departure is None
        assert offer.return_stops is None
