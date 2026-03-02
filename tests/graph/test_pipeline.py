"""Tests for pipeline router functions, _increment_retry, and the errors reducer."""

from __future__ import annotations

from src.app.graph.pipeline import (
    _increment_retry,
    _should_retry_suggest,
    _validation_router,
)


class TestValidationRouter:
    def test_valid(self):
        assert _validation_router({"validated": True}) == "valid"

    def test_invalid(self):
        assert _validation_router({"validated": False}) == "invalid"

    def test_missing_defaults_invalid(self):
        assert _validation_router({}) == "invalid"


class TestShouldRetrySuggest:
    def test_retry_when_no_reports_and_under_max(self):
        state = {"destination_reports": [], "suggest_retry_count": 0}
        assert _should_retry_suggest(state) == "retry_suggest"

    def test_continue_when_reports_exist(self, sample_destination_report):
        state = {"destination_reports": [sample_destination_report], "suggest_retry_count": 0}
        assert _should_retry_suggest(state) == "continue"

    def test_continue_when_at_max_retries(self):
        state = {"destination_reports": [], "suggest_retry_count": 1}
        assert _should_retry_suggest(state) == "continue"


class TestIncrementRetry:
    async def test_increments_from_zero(self):
        result = await _increment_retry({"suggest_retry_count": 0})
        assert result["suggest_retry_count"] == 1

    async def test_increments_from_existing(self):
        result = await _increment_retry({"suggest_retry_count": 2})
        assert result["suggest_retry_count"] == 3


class TestErrorsReducer:
    async def test_errors_accumulate_across_nodes(self):
        """Verify that the Annotated[list[str], operator.add] reducer works
        by running a minimal 2-node graph."""
        from langgraph.graph import END, StateGraph
        from src.app.graph.state import TravelState

        async def node_a(state: TravelState) -> dict:
            return {"errors": ["error from A"]}

        async def node_b(state: TravelState) -> dict:
            return {"errors": ["error from B"]}

        g = StateGraph(TravelState)
        g.add_node("a", node_a)
        g.add_node("b", node_b)
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
        compiled = g.compile()

        result = await compiled.ainvoke({"errors": []})
        assert result["errors"] == ["error from A", "error from B"]
