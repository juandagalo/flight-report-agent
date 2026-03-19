"""WikiVoyage scraper using the MediaWiki API."""

import logging
import re

import httpx

logger = logging.getLogger(__name__)

WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"


async def fetch_page(
    slug: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, str | None]:
    """Fetch a WikiVoyage page by slug.

    Args:
        slug: WikiVoyage page slug (e.g. "Cancun").
        client: Optional pre-existing httpx.AsyncClient. When provided the
            caller owns the client lifecycle (useful for batch fetches).
            When None a short-lived client is created internally.

    Returns:
        Dict with keys:
          - "title": page title
          - "wikitext": raw wikitext content (None if page not found)
          - "extract": plain-text extract (None if not available)
    """
    params = {
        "action": "parse",
        "page": slug,
        "prop": "wikitext",
        "format": "json",
        "formatversion": "2",
    }
    if client is not None:
        resp = await client.get(WIKIVOYAGE_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    else:
        async with httpx.AsyncClient(timeout=30) as _client:
            resp = await _client.get(WIKIVOYAGE_API, params=params)
            resp.raise_for_status()
            data = resp.json()

    if "error" in data:
        logger.warning("WikiVoyage page not found: %s", slug)
        return {"title": slug, "wikitext": None, "extract": None}

    parse = data.get("parse", {})
    wikitext_field = parse.get("wikitext")
    if isinstance(wikitext_field, dict):
        wikitext = wikitext_field.get("*")
    else:
        wikitext = wikitext_field

    return {
        "title": parse.get("title", slug),
        "wikitext": wikitext,
        "extract": None,  # plain text not available from parse endpoint
    }


def parse_sections(wikitext: str) -> list[dict[str, str]]:
    """Parse wikitext into sections.

    Returns a list of dicts with keys:
      - "heading": section heading (empty string for lead section)
      - "level": heading level (2 for ==, 3 for ===, etc.)
      - "content": section text with wikitext markup stripped

    Strips:
      - [[ ]] wiki links (keeps display text)
      - {{ }} templates
      - HTML tags
      - Multiple consecutive blank lines
    """
    if not wikitext or not wikitext.strip():
        return []

    # Split wikitext by headings. Pattern matches == Heading == through ===== Heading =====
    heading_pattern = re.compile(r"^(={2,5})\s*(.+?)\s*\1\s*$", re.MULTILINE)

    sections: list[dict[str, str]] = []
    last_end = 0
    last_heading = ""
    last_level = "0"

    for match in heading_pattern.finditer(wikitext):
        # Capture content before this heading
        content_before = wikitext[last_end : match.start()]
        stripped = strip_wikitext(content_before)
        if stripped or last_heading or sections:
            sections.append(
                {
                    "heading": last_heading,
                    "level": last_level,
                    "content": stripped,
                }
            )

        equals = match.group(1)
        last_level = str(len(equals))
        last_heading = match.group(2).strip()
        last_end = match.end()

    # Capture remaining content after the last heading
    remaining = wikitext[last_end:]
    stripped = strip_wikitext(remaining)
    if stripped or last_heading:
        sections.append(
            {
                "heading": last_heading,
                "level": last_level,
                "content": stripped,
            }
        )

    return sections


def strip_wikitext(text: str) -> str:
    """Remove wikitext markup, keeping readable plain text."""
    # Remove nested templates {{...}} (handle simple nesting)
    # Repeat to handle nested templates
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Remove category links
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text)
    # Convert [[link|display]] to display, [[link]] to link
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    # Remove external links [http://... display] -> display
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    # Remove bare external links [http://...]
    text = re.sub(r"\[https?://\S+\]", "", text)
    # Remove paired HTML tags and their content (ref, gallery, comment, etc.)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<gallery[^>]*>.*?</gallery>", "", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove remaining HTML tags (including self-closing and with attributes)
    text = re.sub(r"<[^>]+>", "", text)
    # Remove bold/italic markup
    text = re.sub(r"'{2,5}", "", text)
    # Remove bullet/numbered list markers at start of lines (keep content)
    text = re.sub(r"^\*+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
