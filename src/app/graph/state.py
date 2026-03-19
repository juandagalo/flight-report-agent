"""LangGraph shared state definition."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from src.app.schemas import (
    CandidateDestination,
    DestinationReport,
    TravelRequest,
)


class TravelState(TypedDict, total=False):
    """State that flows through every node in the graph."""

    # Input (natural language)
    user_message: str

    # Input (structured)
    request: TravelRequest

    # After intake
    intake_assumptions: list[str]

    # After validation
    validated: bool
    validation_errors: list[str]

    # After destination suggestion
    candidate_destinations: list[CandidateDestination]

    # After flight search
    destination_reports: list[DestinationReport]

    # After enrichment (same list, now with weather + activities)
    enriched: bool

    # After report generation
    report_path: str

    # Retry tracking
    suggest_retry_count: int

    # User identification (for RAG personalization)
    user_id: str

    # General errors (auto-appended via reducer)
    errors: Annotated[list[str], operator.add]
