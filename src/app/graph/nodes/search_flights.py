"""Node: Search flights on Amadeus for each candidate destination."""

from __future__ import annotations

import asyncio
import logging

from src.app.graph.state import TravelState
from src.app.schemas import DestinationReport
from src.app.services.amadeus_client import search_flights

logger = logging.getLogger(__name__)


async def search_flights_node(state: TravelState) -> dict:
    """Query Amadeus for flights to every candidate destination."""

    logger.info("→ Starting SEARCH_FLIGHTS node")

    request = state.get("request")
    if request is None:
        return {
            "destination_reports": [],
            "errors": ["No request found in state"],
        }

    candidates = state.get("candidate_destinations", [])

    if not candidates:
        return {
            "destination_reports": [],
            "errors": [
                "No hay destinos candidatos para buscar vuelos."
            ],
        }

    # Use the first date range for the search
    date_range = request.available_dates[0]
    reports: list[DestinationReport] = []

    for dest in candidates:
        logger.info("Searching flights %s → %s", request.origin, dest.iata_code)
        flights = await asyncio.to_thread(
            search_flights,
            origin=request.origin,
            destination=dest.iata_code,
            departure_date=date_range.date_from.isoformat(),
            return_date=date_range.date_to.isoformat(),
            adults=request.num_people,
            currency=request.currency,
            max_results=3,
        )

        # Filter by budget (per-person price)
        affordable = [
            f for f in flights
            if f.price <= request.max_budget * request.num_people
        ]

        if affordable:
            reports.append(DestinationReport(
                destination=dest,
                flights=affordable,
            ))
            logger.info(
                "  Found %d affordable flights to %s", len(affordable), dest.city
            )
        else:
            logger.info(
                "  No affordable flights to %s (found %d over budget)",
                dest.city, len(flights),
            )

    return {
        "destination_reports": reports,
    }
