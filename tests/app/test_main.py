"""Tests for FastAPI endpoints via httpx AsyncClient."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient
import pytest

from src.app.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ── SSE helpers ──────────────────────────────────────────────────────


def _parse_sse_events(text: str) -> list[dict]:
    """Parse raw SSE text into a list of {"event": ..., "data": ...} dicts."""
    events: list[dict] = []
    current_event = None
    current_data = None
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:"):].strip()
        elif line == "":
            if current_event is not None and current_data is not None:
                try:
                    parsed = json.loads(current_data)
                except json.JSONDecodeError:
                    parsed = current_data
                events.append({"event": current_event, "data": parsed})
            current_event = None
            current_data = None
    # Handle last event if text doesn't end with blank line
    if current_event is not None and current_data is not None:
        try:
            parsed = json.loads(current_data)
        except json.JSONDecodeError:
            parsed = current_data
        events.append({"event": current_event, "data": parsed})
    return events


def _make_astream_chunks(*node_deltas: tuple[str, dict]):
    """Build an async iterator of {node_name: delta} dicts for mocking astream."""
    async def _gen(*_args, **_kwargs):
        for name, delta in node_deltas:
            yield {name: delta}
    return _gen


# ── Health ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        async with client as c:
            resp = await c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── POST /api/chat ───────────────────────────────────────────────────


class TestChatEndpoint:
    async def test_empty_body_422(self, client):
        async with client as c:
            resp = await c.post("/api/chat", json={})
        assert resp.status_code == 422

    @patch("src.app.main.travel_graph")
    async def test_pipeline_exception_500(self, mock_graph, client):
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

        async with client as c:
            resp = await c.post("/api/chat", json={"message": "test"})
        assert resp.status_code == 500


# ── POST /api/chat/stream ───────────────────────────────────────────


class TestChatStreamEndpoint:
    async def test_empty_body_422(self, client):
        async with client as c:
            resp = await c.post("/api/chat/stream", json={})
        assert resp.status_code == 422

    @patch("src.app.main.travel_graph")
    async def test_successful_stream_emits_complete(self, mock_graph, client, tmp_path):
        # Create a real PDF file for the report path check
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")

        mock_graph.astream = _make_astream_chunks(
            ("intake", {"request": {}}),
            ("validate", {"validated": True}),
            ("suggest", {"candidate_destinations": []}),
            ("search_flights", {"destination_reports": []}),
            ("enrich", {"enriched": True}),
            ("generate_report", {"report_path": str(pdf_path)}),
        )

        async with client as c:
            resp = await c.post(
                "/api/chat/stream",
                json={"message": "test request"},
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)

        node_start_events = [e for e in events if e["event"] == "node_start"]
        node_end_events = [e for e in events if e["event"] == "node_end"]
        complete_events = [e for e in events if e["event"] == "complete"]

        assert len(node_start_events) == 6
        assert len(node_end_events) == 6
        assert len(complete_events) == 1
        assert complete_events[0]["data"]["report_url"].startswith("/api/reports/")

    @patch("src.app.main.travel_graph")
    async def test_validation_failure_emits_error(self, mock_graph, client):
        mock_graph.astream = _make_astream_chunks(
            ("intake", {"request": {}}),
            ("validate", {"validated": False, "validation_errors": ["fecha inválida"]}),
        )

        async with client as c:
            resp = await c.post(
                "/api/chat/stream",
                json={"message": "test"},
            )

        events = _parse_sse_events(resp.text)
        error_events = [e for e in events if e["event"] == "error"]

        assert len(error_events) == 1
        assert "validación" in error_events[0]["data"]["message"]
        assert "fecha inválida" in error_events[0]["data"]["errors"]

    @patch("src.app.main.travel_graph")
    async def test_pipeline_exception_emits_error(self, mock_graph, client):
        async def _exploding_stream(*_args, **_kwargs):
            raise RuntimeError("kaboom")
            yield  # noqa: unreachable — makes this an async generator

        mock_graph.astream = _exploding_stream

        async with client as c:
            resp = await c.post(
                "/api/chat/stream",
                json={"message": "test"},
            )

        events = _parse_sse_events(resp.text)
        error_events = [e for e in events if e["event"] == "error"]

        assert len(error_events) == 1
        assert "kaboom" in error_events[0]["data"]["message"]

    @patch("src.app.main.travel_graph")
    async def test_retry_nodes_appear_in_stream(self, mock_graph, client, tmp_path):
        pdf_path = tmp_path / "retry_report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")

        mock_graph.astream = _make_astream_chunks(
            ("intake", {"request": {}}),
            ("validate", {"validated": True}),
            ("suggest", {"candidate_destinations": []}),
            ("search_flights", {"destination_reports": []}),
            ("increment_retry", {"suggest_retry_count": 1}),
            ("suggest", {"candidate_destinations": []}),
            ("search_flights", {"destination_reports": []}),
            ("enrich", {"enriched": True}),
            ("generate_report", {"report_path": str(pdf_path)}),
        )

        async with client as c:
            resp = await c.post(
                "/api/chat/stream",
                json={"message": "test retry"},
            )

        events = _parse_sse_events(resp.text)
        node_names = [e["data"]["node"] for e in events if e["event"] == "node_start"]

        assert "increment_retry" in node_names
        assert node_names.count("suggest") == 2


# ── GET /api/reports/{filename} ──────────────────────────────────────


class TestDownloadReport:
    async def test_nonexistent_file_404(self, client):
        async with client as c:
            resp = await c.get("/api/reports/nonexistent.pdf")
        assert resp.status_code == 404

    async def test_path_traversal_rejected(self, client):
        async with client as c:
            resp = await c.get("/api/reports/..secret.pdf")
        assert resp.status_code == 400

    async def test_non_pdf_rejected(self, client):
        async with client as c:
            resp = await c.get("/api/reports/malicious.txt")
        assert resp.status_code == 400

    @patch("src.app.main.settings")
    async def test_valid_pdf_served(self, mock_settings, client, tmp_path):
        mock_settings.REPORT_OUTPUT_DIR = str(tmp_path)
        pdf_file = tmp_path / "test_report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        async with client as c:
            resp = await c.get("/api/reports/test_report.pdf")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.content == b"%PDF-1.4 fake content"
