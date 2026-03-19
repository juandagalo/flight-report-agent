"""CLI entry point: scrape WikiVoyage -> chunk -> embed -> upsert to Qdrant."""

import asyncio
import logging

import httpx

from src.app.config import settings
from src.app.services.embeddings import embed_texts
from src.app.services.qdrant_client import ensure_all_collections, upsert_points
from src.app.services.scraper.chunker import chunk_sections
from src.app.services.scraper.destinations import DESTINATIONS
from src.app.services.scraper.wikivoyage import WIKIVOYAGE_USER_AGENT, fetch_page, parse_sections

logger = logging.getLogger(__name__)

BATCH_SIZE = 20  # Embed this many chunks at a time


async def ingest_destination(
    dest: dict[str, str],
    client: httpx.AsyncClient | None = None,
) -> int:
    """Scrape, chunk, embed, and upsert one destination.

    Args:
        dest: Destination dict with keys wikivoyage_slug, city, country, iata.
        client: Optional shared httpx.AsyncClient for batch operations.

    Returns the number of chunks ingested.
    """
    slug = dest["wikivoyage_slug"]
    meta = {"city": dest["city"], "country": dest["country"], "iata": dest["iata"]}

    page = await fetch_page(slug, client=client)
    if page["wikitext"] is None:
        logger.warning("Skipping %s: page not found", slug)
        return 0

    sections = parse_sections(page["wikitext"])
    chunks = chunk_sections(sections, meta)

    if not chunks:
        logger.warning("No chunks generated for %s", slug)
        return 0

    # Embed and upsert in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = await embed_texts(texts)
        ids = [c["id"] for c in batch]
        payloads = [
            {k: v for k, v in c.items() if k != "id"}
            for c in batch
        ]
        upsert_points(
            collection_name=settings.QDRANT_COLLECTION_KNOWLEDGE,
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        )

    logger.info("Ingested %d chunks for %s (%s)", len(chunks), dest["city"], dest["iata"])
    return len(chunks)


async def ingest_all() -> dict[str, int]:
    """Ingest all curated destinations.

    Uses a single httpx.AsyncClient for all page fetches to avoid
    creating and destroying a connection per destination.

    Returns dict mapping city -> chunk_count.
    """
    ensure_all_collections()
    results = {}
    failed: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": WIKIVOYAGE_USER_AGENT}) as client:
        for dest in DESTINATIONS:
            try:
                count = await ingest_destination(dest, client=client)
                results[dest["city"]] = count
            except Exception:
                logger.exception("Failed to ingest destination %s (%s)", dest["city"], dest["iata"])
                failed[dest["city"]] = dest["iata"]
    if failed:
        logger.error(
            "Ingestion failed for %d destination(s): %s",
            len(failed),
            ", ".join(f"{city} ({iata})" for city, iata in failed.items()),
        )
    return results


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    )
    logger.info("Starting WikiVoyage ingestion for %d destinations", len(DESTINATIONS))
    results = asyncio.run(ingest_all())
    total = sum(results.values())
    logger.info("Ingestion complete: %d total chunks across %d destinations", total, len(results))
    for city, count in sorted(results.items()):
        logger.info("  %s: %d chunks", city, count)


if __name__ == "__main__":
    main()
