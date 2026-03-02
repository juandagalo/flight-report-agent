"""Tests for amadeus_client — _parse_offers (pure) + SDK mock."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.app.services.amadeus_client import _parse_offers, search_flights


# ── _parse_offers (pure) ──────────────────────────────────────────────


def _raw_offer(
    price: float = 500.0,
    carrier: str = "AV",
    out_segments: int = 1,
    ret_segments: int = 1,
) -> dict:
    """Build a minimal Amadeus-style raw offer dict."""
    def _segments(n: int):
        segs = []
        for _ in range(n):
            segs.append({
                "carrierCode": carrier,
                "departure": {"at": "2026-07-01T08:00:00"},
                "arrival": {"at": "2026-07-01T12:00:00"},
            })
        return segs

    return {
        "price": {"total": str(price)},
        "itineraries": [
            {"duration": "PT4H0M", "segments": _segments(out_segments)},
            {"duration": "PT4H0M", "segments": _segments(ret_segments)},
        ],
    }


class TestParseOffers:
    def test_roundtrip(self):
        offers = _parse_offers([_raw_offer()], "USD")
        assert len(offers) == 1
        assert offers[0].price == 500.0
        assert offers[0].stops == 0
        assert offers[0].return_stops == 0

    def test_one_way(self):
        raw = _raw_offer()
        raw["itineraries"] = raw["itineraries"][:1]  # remove return
        offers = _parse_offers([raw], "USD")
        assert len(offers) == 1
        assert offers[0].return_departure is None

    def test_empty_data(self):
        assert _parse_offers([], "USD") == []

    def test_malformed_produces_empty_defaults(self):
        """Completely malformed data still produces an offer with zero/empty defaults."""
        offers = _parse_offers([{"bad": "data"}], "USD")
        assert len(offers) == 1
        assert offers[0].price == 0.0
        assert offers[0].airline == ""

    def test_carrier_name_from_carriers_dict(self):
        """Carrier name is resolved from the top-level carriers dict."""
        carriers = {"AV": "Avianca"}
        offers = _parse_offers([_raw_offer(carrier="AV")], "USD", carriers=carriers)
        assert offers[0].airline == "Avianca"

    def test_carrier_code_fallback(self):
        """Without a carriers dict, the raw carrier code is used."""
        offers = _parse_offers([_raw_offer(carrier="AV")], "USD")
        assert offers[0].airline == "AV"

    def test_multiple_segments_count_stops(self):
        offers = _parse_offers([_raw_offer(out_segments=3)], "USD")
        assert offers[0].stops == 2


# ── search_flights (SDK mock) ────────────────────────────────────────


def _mock_response(data: list[dict], carriers: dict | None = None) -> MagicMock:
    """Build a mock Amadeus Response with .data and .result."""
    resp = MagicMock()
    resp.data = data
    result = {"data": data}
    if carriers:
        result["dictionaries"] = {"carriers": carriers}
    resp.result = result
    return resp


class TestSearchFlights:
    @patch("src.app.services.amadeus_client._get_client")
    def test_success(self, mock_get_client):
        mock_resp = _mock_response([_raw_offer(price=700)])
        mock_client = MagicMock()
        mock_client.shopping.flight_offers_search.get.return_value = mock_resp
        mock_get_client.return_value = mock_client

        results = search_flights("BOG", "CUN", "2026-07-01", "2026-07-15")
        assert len(results) == 1
        assert results[0].price == 700.0

    @patch("src.app.services.amadeus_client._get_client")
    def test_carrier_name_resolved(self, mock_get_client):
        """Airline name is resolved from the response-level dictionaries."""
        mock_resp = _mock_response(
            [_raw_offer(carrier="AV")],
            carriers={"AV": "Avianca"},
        )
        mock_client = MagicMock()
        mock_client.shopping.flight_offers_search.get.return_value = mock_resp
        mock_get_client.return_value = mock_client

        results = search_flights("BOG", "CUN", "2026-07-01", "2026-07-15")
        assert results[0].airline == "Avianca"

    @patch("src.app.services.amadeus_client.time.sleep")
    @patch("src.app.services.amadeus_client._get_client")
    def test_5xx_retry_then_success(self, mock_get_client, mock_sleep):
        from amadeus import ResponseError

        err = ResponseError(MagicMock())
        err.response = MagicMock(status_code=500)

        mock_resp = _mock_response([_raw_offer()])
        mock_client = MagicMock()
        mock_client.shopping.flight_offers_search.get.side_effect = [err, mock_resp]
        mock_get_client.return_value = mock_client

        results = search_flights("BOG", "CUN", "2026-07-01", "2026-07-15")
        assert len(results) == 1
        mock_sleep.assert_called_once()

    @patch("src.app.services.amadeus_client.time.sleep")
    @patch("src.app.services.amadeus_client._get_client")
    def test_retry_exhaustion(self, mock_get_client, mock_sleep):
        from amadeus import ResponseError

        err = ResponseError(MagicMock())
        err.response = MagicMock(status_code=500)

        mock_client = MagicMock()
        mock_client.shopping.flight_offers_search.get.side_effect = [err, err, err]
        mock_get_client.return_value = mock_client

        results = search_flights("BOG", "CUN", "2026-07-01", "2026-07-15")
        assert results == []

    @patch("src.app.services.amadeus_client._get_client")
    def test_non_5xx_no_retry(self, mock_get_client):
        from amadeus import ResponseError

        err = ResponseError(MagicMock())
        err.response = MagicMock(status_code=400)

        mock_client = MagicMock()
        mock_client.shopping.flight_offers_search.get.side_effect = err
        mock_get_client.return_value = mock_client

        results = search_flights("BOG", "CUN", "2026-07-01", "2026-07-15")
        assert results == []
        assert mock_client.shopping.flight_offers_search.get.call_count == 1
