"""LangGraph shared state definition."""

from __future__ import annotations

from typing import TypedDict

from src.app.schemas import (
    CandidateDestination,
    DestinationReport,
    TravelRequest,
)


class TravelState(TypedDict, total=False):
    """State that flows through every node in the graph."""

    # Input
    request: TravelRequest

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

    # General errors
    errors: list[str]
