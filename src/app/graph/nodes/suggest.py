"""Node: Suggest candidate destinations using the LLM."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.app.graph.state import TravelState
from src.app.prompts.templates import (
    SUGGEST_DESTINATIONS_RETRY,
    SUGGEST_DESTINATIONS_SYSTEM,
    SUGGEST_DESTINATIONS_SYSTEM_RAG,
    SUGGEST_DESTINATIONS_USER,
    SUGGEST_DESTINATIONS_USER_RAG,
)
from src.app.schemas import CandidateDestination
from src.app.services.llm_provider import get_llm
from src.app.services.rag import (
    format_rag_context,
    query_interactions,
    query_travel_knowledge,
)

logger = logging.getLogger(__name__)


class DestinationList(BaseModel):
    """Structured output: list of destinations."""

    destinations: list[CandidateDestination] = Field(
        ..., min_length=1, max_length=10
    )


async def suggest_destinations(state: TravelState) -> dict:
    """Call the LLM to suggest 5-8 destinations matching user preferences."""

    logger.info("-> Starting SUGGEST node")

    request = state.get("request")
    if request is None:
        return {"errors": ["No request found in state"]}

    retry_count = state.get("suggest_retry_count", 0)

    dates_str = ", ".join(
        f"{d.date_from.isoformat()} a {d.date_to.isoformat()}"
        for d in request.available_dates
    )
    activities_str = ", ".join(request.preferred_activities)

    # ── RAG context retrieval (non-fatal) ─────────────────────────────
    rag_context = ""
    interaction_context = ""
    try:
        rag_query = (
            f"{request.preferred_climate} {request.region} "
            f"{' '.join(request.preferred_activities)}"
        )
        knowledge_results = await query_travel_knowledge(rag_query, limit=5)
        rag_context = format_rag_context(
            knowledge_results, label="Destination Knowledge"
        )
        logger.info("  Retrieved %d travel knowledge chunks", len(knowledge_results))

        user_id = state.get("user_id", "")
        interaction_results = await query_interactions(rag_query, user_id, limit=3)
        interaction_context = format_rag_context(
            interaction_results, label="User History"
        )
        logger.info("  Retrieved %d interaction history results", len(interaction_results))
    except Exception as exc:
        logger.warning("RAG retrieval failed, proceeding without context: %s", exc)
        rag_context = ""
        interaction_context = ""

    # ── Build LLM prompt ──────────────────────────────────────────────

    llm = get_llm(temperature=0.7)
    structured_llm = llm.with_structured_output(DestinationList)

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
        system_msg = SUGGEST_DESTINATIONS_SYSTEM
    elif rag_context or interaction_context:
        system_msg = SUGGEST_DESTINATIONS_SYSTEM_RAG
        user_msg = SUGGEST_DESTINATIONS_USER_RAG.format(
            origin=request.origin,
            preferred_climate=request.preferred_climate,
            region=request.region,
            dates=dates_str,
            max_budget=request.max_budget,
            currency=request.currency,
            activities=activities_str,
            num_people=request.num_people,
            rag_context=rag_context,
            interaction_context=interaction_context,
        )
    else:
        system_msg = SUGGEST_DESTINATIONS_SYSTEM
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
            {"role": "system", "content": system_msg},
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
