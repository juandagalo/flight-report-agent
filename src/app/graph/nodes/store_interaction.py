"""Node: Store the completed interaction in Qdrant for future RAG context."""

import hashlib
import logging
from datetime import datetime, timezone

from src.app.config import settings
from src.app.graph.state import TravelState
from src.app.services.embeddings import embed_texts
from src.app.services.qdrant_client import ensure_all_collections, upsert_points

logger = logging.getLogger(__name__)


async def store_interaction_node(state: TravelState) -> dict:
    """Store a summary of this interaction for future personalization.

    Stores: user preferences, chosen destinations, and scores.
    Only stores if user_id is provided and destinations exist.
    """
    user_id = state.get("user_id", "")
    if not user_id:
        logger.info("No user_id -- skipping interaction storage")
        return {}

    request = state.get("request")
    reports = state.get("destination_reports", [])
    if not request or not reports:
        return {}

    try:
        ensure_all_collections()

        # Build a text summary of this interaction
        destinations_summary = ", ".join(
            f"{r.destination.city} ({r.destination.iata_code}, score={r.overall_score})"
            for r in reports
        )
        summary = (
            f"Travel search from {request.origin} to {request.region}. "
            f"Climate: {request.preferred_climate}. "
            f"Activities: {', '.join(request.preferred_activities)}. "
            f"Budget: {request.max_budget} {request.currency}. "
            f"Destinations found: {destinations_summary}."
        )

        # Embed the summary
        vectors = await embed_texts([summary])

        # Deterministic ID from user_id + timestamp
        ts = datetime.now(timezone.utc).isoformat()
        interaction_id = hashlib.sha256(f"{user_id}_{ts}".encode()).hexdigest()[:16]

        upsert_points(
            collection_name=settings.QDRANT_COLLECTION_INTERACTIONS,
            ids=[interaction_id],
            vectors=vectors,
            payloads=[{
                "user_id": user_id,
                "text": summary,
                "origin": request.origin,
                "region": request.region,
                "climate": request.preferred_climate,
                "activities": request.preferred_activities,
                "destinations": [r.destination.iata_code for r in reports],
                "timestamp": ts,
            }],
        )
        logger.info("Stored interaction for user %s", user_id)
    except Exception as exc:
        logger.error("Failed to store interaction: %s", exc)
        # Non-fatal -- do not add to errors

    return {}
