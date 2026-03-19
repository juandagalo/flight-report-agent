"""Text chunker for WikiVoyage sections -- token-aware with overlap."""

import hashlib
import logging

import tiktoken

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    model: str = "gpt-4o",
) -> list[str]:
    """Split text into chunks of at most max_tokens with overlap.

    Strategy:
      1. Split by paragraphs (double newline).
      2. Accumulate paragraphs until adding the next would exceed max_tokens.
      3. Emit the chunk, then start the next chunk with the last overlap_tokens
         worth of text from the previous chunk.
      4. If a single paragraph exceeds max_tokens, split it by sentences.
         If a single sentence exceeds max_tokens, split it by words.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    enc = tiktoken.encoding_for_model(model)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    # Expand paragraphs that are individually too long into sentences
    expanded: list[str] = []
    for para in paragraphs:
        if len(enc.encode(para)) <= max_tokens:
            expanded.append(para)
        else:
            # Split paragraph into sentences
            sentences = _split_sentences(para)
            for sentence in sentences:
                if len(enc.encode(sentence)) <= max_tokens:
                    expanded.append(sentence)
                else:
                    # Split sentence by words as last resort
                    word_chunks = _split_by_words(sentence, max_tokens, enc)
                    expanded.extend(word_chunks)

    # Accumulate expanded segments into chunks
    chunks: list[str] = []
    current_parts: list[str] = []
    current_token_count = 0

    for segment in expanded:
        segment_tokens = len(enc.encode(segment))

        # Check if adding this segment would exceed max_tokens
        # Account for the joining newline
        join_overhead = len(enc.encode("\n\n")) if current_parts else 0
        if current_parts and (current_token_count + join_overhead + segment_tokens) > max_tokens:
            # Emit current chunk
            chunk_text_str = "\n\n".join(current_parts)
            chunks.append(chunk_text_str)

            # Build overlap from the tail of the current chunk
            overlap_parts = _build_overlap(current_parts, overlap_tokens, enc)
            current_parts = overlap_parts
            current_token_count = len(enc.encode("\n\n".join(current_parts))) if current_parts else 0

        current_parts.append(segment)
        current_token_count = len(enc.encode("\n\n".join(current_parts)))

    # Emit remaining content
    if current_parts:
        chunk_text_str = "\n\n".join(current_parts)
        chunks.append(chunk_text_str)

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on period/question/exclamation followed by space."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_words(text: str, max_tokens: int, enc: tiktoken.Encoding) -> list[str]:
    """Split text by words when it exceeds max_tokens."""
    words = text.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_count = 0

    for word in words:
        word_tokens = len(enc.encode(word))
        # +1 for the space between words
        overhead = 1 if current_words else 0
        if current_words and (current_count + overhead + word_tokens) > max_tokens:
            chunks.append(" ".join(current_words))
            current_words = []
            current_count = 0
        current_words.append(word)
        current_count = len(enc.encode(" ".join(current_words)))

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _build_overlap(
    parts: list[str], overlap_tokens: int, enc: tiktoken.Encoding
) -> list[str]:
    """Select trailing parts whose combined tokens are <= overlap_tokens."""
    overlap_parts: list[str] = []
    token_count = 0

    for part in reversed(parts):
        part_tokens = len(enc.encode(part))
        join_overhead = len(enc.encode("\n\n")) if overlap_parts else 0
        if token_count + join_overhead + part_tokens > overlap_tokens:
            break
        overlap_parts.insert(0, part)
        token_count += join_overhead + part_tokens

    return overlap_parts


def chunk_sections(
    sections: list[dict[str, str]],
    destination_meta: dict[str, str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict[str, str]]:
    """Chunk WikiVoyage sections and attach metadata.

    Args:
        sections: Output from parse_sections().
        destination_meta: Dict with city, country, iata keys.
        max_tokens: Max tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks.

    Returns:
        List of dicts with keys:
          - "id": deterministic hash of content
          - "text": chunk text
          - "heading": section heading this chunk belongs to
          - "city": from destination_meta
          - "country": from destination_meta
          - "iata": from destination_meta
    """
    results = []
    for section in sections:
        heading = section["heading"]
        content = section["content"]
        if not content.strip():
            continue

        # Prepend heading for context
        full_text = f"{heading}\n\n{content}" if heading else content
        chunks = chunk_text(full_text, max_tokens, overlap_tokens)

        for idx, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{heading}:{idx}:{chunk}".encode()).hexdigest()[:16]
            results.append(
                {
                    "id": f"{destination_meta['iata']}_{chunk_id}",
                    "text": chunk,
                    "heading": heading,
                    "city": destination_meta["city"],
                    "country": destination_meta["country"],
                    "iata": destination_meta["iata"],
                }
            )
    return results
