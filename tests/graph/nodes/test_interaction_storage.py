"""Tests for the store_interaction node."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from src.app.graph.nodes.store_interaction import store_interaction_node
from src.app.schemas import (
    CandidateDestination,
    DestinationReport,
    FlightOffer,
)


def _make_report(city: str = "Cancun", iata: str = "CUN", score: int = 75):
    return DestinationReport(
        destination=CandidateDestination(
            city=city, iata_code=iata, country="Mexico",
            climate_match=85, activity_match=70,
        ),
        flights=[FlightOffer(price=800.0, stops=0)],
        overall_score=score,
    )


class TestStoreInteractionNode:
    @patch("src.app.graph.nodes.store_interaction.upsert_points")
    @patch("src.app.graph.nodes.store_interaction.embed_texts", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.store_interaction.ensure_all_collections")
    async def test_stores_interaction_with_user_id(
        self,
        mock_ensure,
        mock_embed,
        mock_upsert,
        sample_travel_request,
    ):
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        report = _make_report()
        state = {
            "user_id": "user-42",
            "request": sample_travel_request,
            "destination_reports": [report],
        }
        result = await store_interaction_node(state)

        assert result == {}
        mock_ensure.assert_called_once()
        mock_embed.assert_called_once()

        # Verify upsert was called with correct arguments
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "interactions"
        assert len(call_kwargs.kwargs["ids"]) == 1
        assert call_kwargs.kwargs["vectors"] == [[0.1, 0.2, 0.3]]

        payload = call_kwargs.kwargs["payloads"][0]
        assert payload["user_id"] == "user-42"
        assert "Travel search" in payload["text"]
        assert payload["origin"] == "BOG"
        assert "CUN" in payload["destinations"]
        assert "timestamp" in payload

    async def test_skips_storage_without_user_id(self, sample_travel_request):
        report = _make_report()
        state = {
            "user_id": "",
            "request": sample_travel_request,
            "destination_reports": [report],
        }
        result = await store_interaction_node(state)
        assert result == {}

    async def test_skips_storage_without_reports(self, sample_travel_request):
        state = {
            "user_id": "user-42",
            "request": sample_travel_request,
            "destination_reports": [],
        }
        result = await store_interaction_node(state)
        assert result == {}

    async def test_skips_storage_without_request(self):
        report = _make_report()
        state = {
            "user_id": "user-42",
            "destination_reports": [report],
        }
        result = await store_interaction_node(state)
        assert result == {}

    @patch("src.app.graph.nodes.store_interaction.upsert_points")
    @patch("src.app.graph.nodes.store_interaction.embed_texts", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.store_interaction.ensure_all_collections")
    async def test_storage_failure_non_fatal(
        self,
        mock_ensure,
        mock_embed,
        mock_upsert,
        sample_travel_request,
    ):
        mock_embed.return_value = [[0.1, 0.2]]
        mock_upsert.side_effect = RuntimeError("Qdrant write failed")

        report = _make_report()
        state = {
            "user_id": "user-42",
            "request": sample_travel_request,
            "destination_reports": [report],
        }
        # Should NOT raise -- errors are logged but not propagated
        result = await store_interaction_node(state)
        assert result == {}

    @patch("src.app.graph.nodes.store_interaction.upsert_points")
    @patch("src.app.graph.nodes.store_interaction.embed_texts", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.store_interaction.ensure_all_collections")
    async def test_embed_failure_non_fatal(
        self,
        mock_ensure,
        mock_embed,
        mock_upsert,
        sample_travel_request,
    ):
        mock_embed.side_effect = RuntimeError("Embedding service down")

        report = _make_report()
        state = {
            "user_id": "user-42",
            "request": sample_travel_request,
            "destination_reports": [report],
        }
        result = await store_interaction_node(state)
        assert result == {}
        mock_upsert.assert_not_called()

    @patch("src.app.graph.nodes.store_interaction.upsert_points")
    @patch("src.app.graph.nodes.store_interaction.embed_texts", new_callable=AsyncMock)
    @patch("src.app.graph.nodes.store_interaction.ensure_all_collections")
    async def test_stores_multiple_destinations(
        self,
        mock_ensure,
        mock_embed,
        mock_upsert,
        sample_travel_request,
    ):
        mock_embed.return_value = [[0.1, 0.2]]

        reports = [
            _make_report("Cancun", "CUN", 80),
            _make_report("Punta Cana", "PUJ", 75),
        ]
        state = {
            "user_id": "user-99",
            "request": sample_travel_request,
            "destination_reports": reports,
        }
        result = await store_interaction_node(state)

        assert result == {}
        payload = mock_upsert.call_args.kwargs["payloads"][0]
        assert "CUN" in payload["destinations"]
        assert "PUJ" in payload["destinations"]
        assert "Cancun" in payload["text"]
        assert "Punta Cana" in payload["text"]
