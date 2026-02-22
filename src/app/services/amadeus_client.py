"""Amadeus API client wrapper."""

from __future__ import annotations

import logging
from typing import Any

from amadeus import Client, ResponseError

from src.app.config import settings
from src.app.schemas import FlightOffer

logger = logging.getLogger(__name__)


def _get_client() -> Client:
    """Create an Amadeus client from settings."""
    hostname = "test" if settings.AMADEUS_ENV == "test" else "production"
    return Client(
        client_id=settings.AMADEUS_CLIENT_ID,
        client_secret=settings.AMADEUS_CLIENT_SECRET,
        hostname=hostname,
        logger=logger,
    )


def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    adults: int = 1,
    currency: str = "USD",
    max_results: int = 3,
) -> list[FlightOffer]:
    """Search round-trip flights via Amadeus Flight Offers Search API.

    Returns up to *max_results* cheapest offers.
    """
    client = _get_client()
    try:
        response = client.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            returnDate=return_date,
            adults=adults,
            currencyCode=currency,
            max=max_results,
        )
        return _parse_offers(response.data, currency)
    except ResponseError as err:
        logger.error("Amadeus API error for %s→%s: %s", origin, destination, err)
        return []
    except Exception as err:
        logger.error("Unexpected error searching flights: %s", err)
        return []


def _parse_offers(data: list[dict[str, Any]], currency: str) -> list[FlightOffer]:
    """Parse raw Amadeus JSON into FlightOffer models."""
    offers: list[FlightOffer] = []
    for item in data:
        try:
            price = float(item.get("price", {}).get("total", 0))
            itineraries = item.get("itineraries", [])

            # Outbound
            outbound = itineraries[0] if itineraries else {}
            out_segments = outbound.get("segments", [])
            out_first = out_segments[0] if out_segments else {}
            out_last = out_segments[-1] if out_segments else {}

            # Return
            ret = itineraries[1] if len(itineraries) > 1 else {}
            ret_segments = ret.get("segments", [])
            ret_first = ret_segments[0] if ret_segments else {}
            ret_last = ret_segments[-1] if ret_segments else {}

            # Carrier codes → airline name (simplified)
            carrier = out_first.get("carrierCode", "")
            # Try to get the airline name from dictionaries
            carriers_dict = item.get("dictionaries", {}).get("carriers", {})
            airline_name = carriers_dict.get(carrier, carrier)

            offer = FlightOffer(
                airline=airline_name,
                price=price,
                currency=currency,
                departure=out_first.get("departure", {}).get("at", ""),
                arrival=out_last.get("arrival", {}).get("at", ""),
                duration=outbound.get("duration", ""),
                stops=max(0, len(out_segments) - 1),
                return_departure=ret_first.get("departure", {}).get("at", "") or None,
                return_arrival=ret_last.get("arrival", {}).get("at", "") or None,
                return_duration=ret.get("duration", "") or None,
                return_stops=max(0, len(ret_segments) - 1) if ret_segments else None,
            )
            offers.append(offer)
        except (IndexError, KeyError, TypeError) as exc:
            logger.warning("Skipping malformed offer: %s", exc)
    return offers
