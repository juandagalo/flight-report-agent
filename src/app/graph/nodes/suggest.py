"""Node: Suggest candidate destinations using the LLM."""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.app.config import settings
from src.app.graph.state import TravelState
from src.app.prompts.templates import (
    SUGGEST_DESTINATIONS_RETRY,
    SUGGEST_DESTINATIONS_SYSTEM,
    SUGGEST_DESTINATIONS_USER,
)
from src.app.schemas import CandidateDestination

logger = logging.getLogger(__name__)


class DestinationList(BaseModel):
    """Structured output: list of destinations."""

    destinations: list[CandidateDestination] = Field(
        ..., min_length=1, max_length=10
    )


async def suggest_destinations(state: TravelState) -> dict:
    """Call the LLM to suggest 5-8 destinations matching user preferences."""

    logger.info("→ Starting SUGGEST node")

    request = state.get("request")
    if request is None:
        return {"errors": ["No request found in state"]}

    retry_count = state.get("suggest_retry_count", 0)

    dates_str = ", ".join(
        f"{d.date_from.isoformat()} a {d.date_to.isoformat()}"
        for d in request.available_dates
    )
    activities_str = ", ".join(request.preferred_activities)

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
    )
    structured_llm = llm.with_structured_output(DestinationList, method="json_mode")

    if retry_count > 0 and state.get("candidate_destinations"):
        # Retry with exclusion of previous destinations
        prev = ", ".join(
            f"{d.city} ({d.iata_code})" for d in state.get("candidate_destinations", [])
        )
        user_msg = SUGGEST_DESTINATIONS_RETRY.format(
            previous_destinations=prev,
            origin=request.origin,
            preferred_climate=request.preferred_climate,
            activities=activities_str,
            region=request.region,
        )
    else:
        user_msg = SUGGEST_DESTINATIONS_USER.format(
            origin=request.origin,
            preferred_climate=request.preferred_climate,
            region=request.region,
            dates=dates_str,
            max_budget=request.max_budget,
            currency=request.currency,
            activities=activities_str,
            num_people=request.num_people,
        )

    try:
        result: DestinationList = await structured_llm.ainvoke([
            {"role": "system", "content": SUGGEST_DESTINATIONS_SYSTEM},
            {"role": "user", "content": user_msg},
        ])
        destinations = [
            CandidateDestination(**d.model_dump()) for d in result.destinations
        ]
        logger.info(
            "LLM suggested %d destinations: %s",
            len(destinations),
            [d.iata_code for d in destinations],
        )
        return {
            "candidate_destinations": destinations,
        }
    except Exception as exc:
        logger.error("LLM suggestion failed: %s", exc)
        return {
            "candidate_destinations": [],
            "errors": [f"Error al sugerir destinos: {exc}"],
        }
