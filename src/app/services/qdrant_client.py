"""Qdrant vector store client -- embedded mode, disk-persisted."""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.app.config import settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None

# Namespace UUID for deterministic ID generation from arbitrary strings.
_UUID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _to_uuid(string_id: str) -> str:
    """Convert an arbitrary string ID to a deterministic UUID string.

    Qdrant's embedded (local) mode requires string IDs to be valid UUIDs.
    This uses UUID5 with a fixed namespace so the same input always produces
    the same UUID.
    """
    return str(uuid.uuid5(_UUID_NAMESPACE, string_id))


def get_client() -> QdrantClient:
    """Return a singleton QdrantClient in embedded (local disk) mode."""
    global _client
    if _client is None:
        _client = QdrantClient(path=settings.QDRANT_PATH)
    return _client


def ensure_collection(name: str, vector_size: int | None = None) -> None:
    """Create a collection if it does not already exist.

    Args:
        name: Collection name.
        vector_size: Embedding vector dimension. Defaults to settings.EMBEDDING_DIMENSION.
    """
    client = get_client()
    size = vector_size or settings.EMBEDDING_DIMENSION
    collections = [c.name for c in client.get_collections().collections]
    if name not in collections:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s (dim=%d)", name, size)


def ensure_all_collections() -> None:
    """Create both travel_knowledge and interactions collections if missing."""
    ensure_collection(settings.QDRANT_COLLECTION_KNOWLEDGE)
    ensure_collection(settings.QDRANT_COLLECTION_INTERACTIONS)


def upsert_points(
    collection_name: str,
    ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
) -> None:
    """Insert or update points in a collection.

    Args:
        collection_name: Target collection.
        ids: Unique string IDs for each point.
        vectors: Embedding vectors, one per point.
        payloads: Metadata dicts, one per point.
    """
    client = get_client()
    points = [
        PointStruct(id=_to_uuid(uid), vector=vec, payload=pay)
        for uid, vec, pay in zip(ids, vectors, payloads, strict=True)
    ]
    client.upsert(collection_name=collection_name, points=points)


def search(
    collection_name: str,
    query_vector: list[float],
    limit: int = 5,
    score_threshold: float = 0.0,
    filter_conditions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Search for similar vectors in a collection.

    Args:
        collection_name: Collection to search.
        query_vector: Query embedding vector.
        limit: Max number of results.
        score_threshold: Minimum similarity score (0.0 to 1.0).
        filter_conditions: Optional dict of field->value for exact-match filtering.

    Returns:
        List of dicts with keys: id, score, payload.
    """
    client = get_client()
    query_filter = None
    if filter_conditions:
        must = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_conditions.items()
        ]
        query_filter = Filter(must=must)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )
    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in results.points
    ]


def delete_collection(name: str) -> None:
    """Delete a collection. Used in tests and reset operations."""
    client = get_client()
    client.delete_collection(collection_name=name)
    logger.info("Deleted Qdrant collection: %s", name)


def reset_client() -> None:
    """Close and reset the singleton client. Used in tests."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
