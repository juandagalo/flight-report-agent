"""Node: Enrich destination reports with weather and activity data."""

from __future__ import annotations

import asyncio
import logging

from langchain_openai import ChatOpenAI

from src.app.config import settings
from src.app.graph.state import TravelState
from src.app.prompts.templates import ENRICH_ACTIVITIES_SYSTEM, ENRICH_ACTIVITIES_USER
from src.app.services.weather_client import get_weather

logger = logging.getLogger(__name__)


def _compute_score(report, budget: float) -> int:
    """Compute weighted overall score for a destination report."""
    climate = report.destination.climate_match
    activity = report.destination.activity_match

    # Price factor: lower price → higher score
    if report.flights:
        min_price = min(f.price for f in report.flights)
        price_ratio = max(0, 1 - (min_price / budget)) if budget > 0 else 0
        price_score = int(price_ratio * 100)
    else:
        price_score = 0

    # Stops factor: fewer stops → higher score
    if report.flights:
        min_stops = min(f.stops for f in report.flights)
        stops_score = max(0, 100 - min_stops * 30)
    else:
        stops_score = 0

    # Weighted combination
    return int(
        climate * 0.30
        + activity * 0.30
        + price_score * 0.25
        + stops_score * 0.15
    )


async def enrich_data(state: TravelState) -> dict:
    """Add weather info and activity recommendations to each destination."""

    logger.info("→ Starting ENRICH node")

    request = state.get("request")

    if request is None:
        return {"errors": ["No request found in state"]}

    reports = state.get("destination_reports", [])

    if not reports:
        return {"enriched": True}

    date_range = request.available_dates[0]
    activities_str = ", ".join(request.preferred_activities)
    dates_str = f"{date_range.date_from.isoformat()} a {date_range.date_to.isoformat()}"

    # ── Weather (async, gathered concurrently) ────────────────────────

    weather_results = await asyncio.gather(*(
        get_weather(
            iata_code=r.destination.iata_code,
            date_from=date_range.date_from,
            date_to=date_range.date_to,
        )
        for r in reports
    ))

    # ── Activities (LLM, async) ───────────────────────────────────────

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
    )

    activity_results: list[str] = []
    for report in reports:
        try:
            user_msg = ENRICH_ACTIVITIES_USER.format(
                city=report.destination.city,
                country=report.destination.country,
                activities=activities_str,
                dates=dates_str,
            )

            resp = await llm.ainvoke([
                {"role": "system", "content": ENRICH_ACTIVITIES_SYSTEM},
                {"role": "user", "content": user_msg},
            ])
            activity_results.append(resp.content)
        except Exception as exc:
            logger.error(
                "Activity enrichment failed for %s: %s",
                report.destination.city, exc,
            )
            activity_results.append("Información de actividades no disponible.")

    # ── Build updated reports (immutable via model_copy) ──────────────

    budget = request.max_budget * request.num_people
    enriched_reports = []
    for i, report in enumerate(reports):
        updated = report.model_copy(update={
            "weather": weather_results[i],
            "activities_description": activity_results[i],
        })
        updated.overall_score = _compute_score(updated, budget)
        enriched_reports.append(updated)

    return {
        "destination_reports": enriched_reports,
        "enriched": True,
    }
