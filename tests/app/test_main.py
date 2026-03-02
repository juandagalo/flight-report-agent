"""Tests for FastAPI endpoints via httpx AsyncClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient
import pytest

from src.app.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        async with client as c:
            resp = await c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


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
