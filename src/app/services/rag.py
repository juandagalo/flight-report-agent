"""RAG context retrieval helpers for pipeline nodes."""

import logging
import re
from typing import Any

from src.app.config import settings
from src.app.services.embeddings import embed_query
from src.app.services.qdrant_client import ensure_all_collections, search

logger = logging.getLogger(__name__)


async def query_travel_knowledge(
    query_text: str,
    limit: int = 5,
    destination_iata: str | None = None,
) -> list[dict[str, Any]]:
    """Search travel_knowledge collection for relevant destination info.

    Args:
        query_text: Natural language query to embed and search.
        limit: Max results to return.
        destination_iata: Optional IATA filter to scope results to one destination.

    Returns:
        List of result dicts with id, score, payload (text, heading, city, etc.)
    """
    ensure_all_collections()
    vector = await embed_query(query_text)
    filters = {"iata": destination_iata} if destination_iata else None
    return search(
        collection_name=settings.QDRANT_COLLECTION_KNOWLEDGE,
        query_vector=vector,
        limit=limit,
        score_threshold=0.3,
        filter_conditions=filters,
    )


async def query_interactions(
    query_text: str,
    user_id: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Search interactions collection for past user interactions.

    Args:
        query_text: Query to embed and search.
        user_id: User identifier to filter by.
        limit: Max results.

    Returns:
        List of result dicts.
    """
    if not user_id:
        return []

    ensure_all_collections()
    vector = await embed_query(query_text)
    return search(
        collection_name=settings.QDRANT_COLLECTION_INTERACTIONS,
        query_vector=vector,
        limit=limit,
        score_threshold=0.3,
        filter_conditions={"user_id": user_id},
    )


_INJECTION_PATTERN = re.compile(
    r"^\s*("
    r"IGNORE|SYSTEM:|You are|INSTRUCTION|ADMIN:|OVERRIDE|"
    r"<\|im_start\|>|<\|system\|>|<<SYS>>|### Instruction"
    r")",
    re.IGNORECASE,
)


def _sanitize_rag_text(text: str) -> str:
    """Strip lines that look like prompt-injection attempts.

    Lines starting with common injection patterns (IGNORE, SYSTEM:, You are,
    INSTRUCTION, etc.) are removed so that RAG context cannot hijack the LLM.
    """
    clean_lines = [
        line
        for line in text.splitlines()
        if not _INJECTION_PATTERN.match(line)
    ]
    return "\n".join(clean_lines)


def format_rag_context(results: list[dict[str, Any]], label: str = "Context") -> str:
    """Format RAG search results into a text block for LLM prompt inclusion.

    Returns empty string if no results.
    """
    if not results:
        return ""

    lines = [f"--- {label} ---"]
    for r in results:
        payload = r.get("payload", {})
        text = _sanitize_rag_text(payload.get("text", ""))
        city = payload.get("city", "")
        heading = payload.get("heading", "")
        source = f" [{city} - {heading}]" if city else ""
        lines.append(f"- {text[:500]}{source}")
    lines.append(f"--- End {label} ---")
    return "\n".join(lines)
