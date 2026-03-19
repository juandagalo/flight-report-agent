"""Tests for the Qdrant vector store client service."""

import pytest

from src.app.services.qdrant_client import (
    _to_uuid,
    delete_collection,
    ensure_all_collections,
    ensure_collection,
    get_client,
    reset_client,
    search,
    upsert_points,
)


@pytest.fixture(autouse=True)
def _qdrant_tmp(tmp_path, monkeypatch):
    """Point Qdrant to a temp directory and reset client between tests."""
    monkeypatch.setattr(
        "src.app.services.qdrant_client.settings.QDRANT_PATH",
        str(tmp_path / "qdrant"),
    )
    monkeypatch.setattr(
        "src.app.services.qdrant_client.settings.EMBEDDING_DIMENSION",
        4,
    )
    monkeypatch.setattr(
        "src.app.services.qdrant_client.settings.QDRANT_COLLECTION_KNOWLEDGE",
        "travel_knowledge",
    )
    monkeypatch.setattr(
        "src.app.services.qdrant_client.settings.QDRANT_COLLECTION_INTERACTIONS",
        "interactions",
    )
    reset_client()
    yield
    reset_client()


class TestEnsureCollection:
    """Tests for ensure_collection."""

    def test_ensure_collection_creates_new(self):
        """ensure_collection creates a collection that can be listed."""
        ensure_collection("test_xyz", vector_size=4)
        client = get_client()
        names = [c.name for c in client.get_collections().collections]
        assert "test_xyz" in names

    def test_ensure_collection_idempotent(self):
        """Calling ensure_collection twice does not raise and keeps one collection."""
        ensure_collection("test_xyz", vector_size=4)
        ensure_collection("test_xyz", vector_size=4)
        client = get_client()
        names = [c.name for c in client.get_collections().collections]
        assert names.count("test_xyz") == 1


class TestUpsertAndSearch:
    """Tests for upsert_points and search."""

    def test_upsert_and_search(self):
        """Upserting 3 points and searching returns the closest match first."""
        ensure_collection("test_search", vector_size=4)
        upsert_points(
            "test_search",
            ids=["p1", "p2", "p3"],
            vectors=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            payloads=[
                {"name": "alpha"},
                {"name": "beta"},
                {"name": "gamma"},
            ],
        )
        results = search("test_search", query_vector=[1.0, 0.1, 0.0, 0.0], limit=3)
        assert len(results) == 3
        # First result should be the closest to [1.0, 0.1, 0.0, 0.0]
        assert results[0]["payload"]["name"] == "alpha"
        assert results[0]["score"] > results[1]["score"]

    def test_search_with_filter(self):
        """Search with filter_conditions returns only matching points."""
        ensure_collection("test_filter", vector_size=4)
        upsert_points(
            "test_filter",
            ids=["f1", "f2", "f3"],
            vectors=[
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
            ],
            payloads=[
                {"destination": "CUN"},
                {"destination": "BCN"},
                {"destination": "CUN"},
            ],
        )
        results = search(
            "test_filter",
            query_vector=[1.0, 0.0, 0.0, 0.0],
            limit=5,
            filter_conditions={"destination": "CUN"},
        )
        assert len(results) == 2
        for r in results:
            assert r["payload"]["destination"] == "CUN"

    def test_search_score_threshold(self):
        """Search with a very high score threshold returns no results."""
        ensure_collection("test_threshold", vector_size=4)
        upsert_points(
            "test_threshold",
            ids=["t1"],
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            payloads=[{"name": "only"}],
        )
        # Query with an orthogonal vector and high threshold
        results = search(
            "test_threshold",
            query_vector=[0.0, 0.0, 0.0, 1.0],
            limit=5,
            score_threshold=0.99,
        )
        assert len(results) == 0

    def test_upsert_overwrites_existing(self):
        """Upserting with the same ID overwrites the payload."""
        ensure_collection("test_overwrite", vector_size=4)
        upsert_points(
            "test_overwrite",
            ids=["ow1"],
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            payloads=[{"version": "v1"}],
        )
        # Overwrite with same ID but new payload
        upsert_points(
            "test_overwrite",
            ids=["ow1"],
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            payloads=[{"version": "v2"}],
        )
        results = search(
            "test_overwrite",
            query_vector=[1.0, 0.0, 0.0, 0.0],
            limit=1,
        )
        assert len(results) == 1
        assert results[0]["payload"]["version"] == "v2"


class TestDeleteCollection:
    """Tests for delete_collection."""

    def test_delete_collection(self):
        """Creating and then deleting a collection removes it from the listing."""
        ensure_collection("test_delete", vector_size=4)
        client = get_client()
        names_before = [c.name for c in client.get_collections().collections]
        assert "test_delete" in names_before

        delete_collection("test_delete")
        names_after = [c.name for c in client.get_collections().collections]
        assert "test_delete" not in names_after


class TestEnsureAllCollections:
    """Tests for ensure_all_collections."""

    def test_ensure_all_collections(self):
        """ensure_all_collections creates both travel_knowledge and interactions."""
        ensure_all_collections()
        client = get_client()
        names = [c.name for c in client.get_collections().collections]
        assert "travel_knowledge" in names
        assert "interactions" in names


class TestToUuid:
    """Tests for the _to_uuid helper."""

    def test_deterministic(self):
        """Same input always produces the same UUID."""
        assert _to_uuid("test_id") == _to_uuid("test_id")

    def test_different_inputs_produce_different_uuids(self):
        """Different inputs produce different UUIDs."""
        assert _to_uuid("id_a") != _to_uuid("id_b")
