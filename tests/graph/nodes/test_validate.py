"""Tests for the validate node — pure logic, no mocks needed."""

from __future__ import annotations

from datetime import date

import pytest

from src.app.schemas import DateRange, TravelRequest
from src.app.graph.nodes.validate import validate_input


def _make_request(**overrides) -> TravelRequest:
    defaults = dict(
        preferred_climate="tropical",
        region="Caribe",
        available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
        max_budget=1500.0,
        origin="BOG",
        preferred_activities=["playa"],
        num_people=2,
    )
    defaults.update(overrides)
    return TravelRequest(**defaults)


class TestValidateInput:
    async def test_valid_request_passes(self):
        state = {"request": _make_request()}
        result = await validate_input(state)
        assert result["validated"] is True
        assert result["validation_errors"] == []

    async def test_missing_request(self):
        result = await validate_input({})
        assert result["validated"] is False
        assert "No request found in state" in result["errors"]

    async def test_reversed_dates(self):
        req = _make_request(
            available_dates=[DateRange(date_from=date(2026, 7, 15), date_to=date(2026, 7, 1))]
        )
        result = await validate_input({"request": req})
        assert result["validated"] is False
        assert any("posterior" in e for e in result["validation_errors"])

    async def test_same_dates_rejected(self):
        req = _make_request(
            available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 1))]
        )
        result = await validate_input({"request": req})
        assert result["validated"] is False

    async def test_unknown_iata_still_passes(self):
        """An unknown 3-char IATA code logs a warning but passes validation."""
        req = _make_request(origin="XYZ")
        result = await validate_input({"request": req})
        assert result["validated"] is True

    async def test_multiple_errors_collected(self):
        req = _make_request(
            available_dates=[DateRange(date_from=date(2026, 7, 15), date_to=date(2026, 7, 1))],
        )
        result = await validate_input({"request": req})
        assert result["validated"] is False
        assert len(result["validation_errors"]) >= 1
