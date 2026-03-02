"""LangGraph pipeline: wires all nodes into a StateGraph."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.app.graph.nodes.enrich import enrich_data
from src.app.graph.nodes.intake import intake_node
from src.app.graph.nodes.report import generate_report_node
from src.app.graph.nodes.search_flights import search_flights_node
from src.app.graph.nodes.suggest import suggest_destinations
from src.app.graph.nodes.validate import validate_input
from src.app.graph.state import TravelState

logger = logging.getLogger(__name__)

MAX_SUGGEST_RETRIES = 1


def _should_retry_suggest(state: TravelState) -> str:
    """Decide whether to retry destination suggestion or proceed."""

    reports = state.get("destination_reports", [])
    retry_count = state.get("suggest_retry_count", 0)

    if len(reports) == 0 and retry_count < MAX_SUGGEST_RETRIES:
        logger.info("No flights found – retrying suggestion (attempt %d)", retry_count + 1)
        return "retry_suggest"
    return "continue"


async def _increment_retry(state: TravelState) -> dict:
    """Bump the retry counter before re-suggesting."""

    logger.info("→ Starting INCREMENT_RETRY node")

    return {
        "suggest_retry_count": state.get("suggest_retry_count", 0) + 1,
    }


def _validation_router(state: TravelState) -> str:
    """Route based on validation result."""
    if state.get("validated", False):
        return "valid"
    return "invalid"


def build_graph():
    """Construct and compile the travel recommendation graph.

    Returns:
        Compiled LangGraph ready to invoke with TravelState.
    """
    graph = StateGraph(TravelState)

    # ── Add nodes ────────────────────────────────────────────────────
    graph.add_node("intake", intake_node)
    graph.add_node("validate", validate_input)
    graph.add_node("suggest", suggest_destinations)
    graph.add_node("search_flights", search_flights_node)
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("enrich", enrich_data)
    graph.add_node("generate_report", generate_report_node)

    # ── Set entry point ──────────────────────────────────────────────
    graph.set_entry_point("intake")

    graph.add_edge("intake", "validate")

    # ── Edges ────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "validate",
        _validation_router,
        {
            "valid": "suggest",
            "invalid": END,
        },
    )

    graph.add_edge("suggest", "search_flights")

    graph.add_conditional_edges(
        "search_flights",
        _should_retry_suggest,
        {
            "retry_suggest": "increment_retry",
            "continue": "enrich",
        },
    )

    graph.add_edge("increment_retry", "suggest")
    graph.add_edge("enrich", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# Pre-compiled graph instance
travel_graph = build_graph()
