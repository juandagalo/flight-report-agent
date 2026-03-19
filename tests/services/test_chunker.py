"""Tests for the chunker -- token-aware text splitting."""

from __future__ import annotations

import re

from src.app.services.scraper.chunker import (
    chunk_sections,
    chunk_text,
    count_tokens,
)


class TestCountTokens:
    def test_count_tokens_positive(self):
        result = count_tokens("Hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty_string(self):
        assert count_tokens("") == 0

    def test_count_tokens_longer_text(self):
        short = count_tokens("hi")
        long = count_tokens("This is a much longer sentence with many words in it.")
        assert long > short


class TestChunkText:
    def test_chunk_text_short_input(self):
        text = "This is short."
        chunks = chunk_text(text, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_text_splits_at_paragraph_boundary(self):
        # Build multiple paragraphs that collectively exceed max_tokens
        paragraphs = [f"Paragraph number {i} with some extra words to fill up tokens." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)

        assert len(chunks) > 1
        # Each chunk should not exceed max_tokens (approximately -- overlap may cause minor variance)
        for chunk in chunks:
            tokens = count_tokens(chunk)
            # Allow some tolerance for boundary effects
            assert tokens <= 60, f"Chunk has {tokens} tokens, expected <= 60"

    def test_chunk_text_overlap(self):
        # Create text that must be split into at least 2 chunks
        paragraphs = [f"Unique paragraph {i} with content about topic {i}." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, max_tokens=50, overlap_tokens=20)

        assert len(chunks) >= 2
        # Verify overlap: the end of chunk N should share some text with the start of chunk N+1
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            # Get last paragraph of current chunk
            current_paras = current_chunk.split("\n\n")
            next_paras = next_chunk.split("\n\n")
            # With overlap, some paragraphs from the end of the current chunk
            # should appear at the start of the next chunk
            # Check if there is any shared content
            shared = set(current_paras) & set(next_paras)
            assert len(shared) > 0, "Expected overlapping content between consecutive chunks"

    def test_chunk_text_single_long_paragraph(self):
        # A single very long paragraph that exceeds max_tokens
        long_text = " ".join([f"word{i}" for i in range(200)])
        chunks = chunk_text(long_text, max_tokens=50, overlap_tokens=5)

        assert len(chunks) > 1
        # All original words should appear across all chunks
        all_text = " ".join(chunks)
        assert "word0" in all_text
        assert "word199" in all_text

    def test_chunk_text_preserves_all_content(self):
        paragraphs = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, max_tokens=500)

        # With large max_tokens, should return single chunk
        assert len(chunks) == 1
        for para in paragraphs:
            assert para in chunks[0]


class TestChunkSections:
    def test_chunk_sections_attaches_metadata(self):
        sections = [
            {"heading": "See", "level": "2", "content": "Beautiful beaches and ancient ruins."},
        ]
        meta = {"city": "Cancun", "country": "Mexico", "iata": "CUN"}
        result = chunk_sections(sections, meta)

        assert len(result) >= 1
        chunk = result[0]
        assert chunk["city"] == "Cancun"
        assert chunk["country"] == "Mexico"
        assert chunk["iata"] == "CUN"
        assert chunk["heading"] == "See"
        assert "beaches" in chunk["text"].lower() or "Beautiful" in chunk["text"]

    def test_chunk_sections_deterministic_ids(self):
        sections = [
            {"heading": "See", "level": "2", "content": "Beaches and ruins."},
        ]
        meta = {"city": "Cancun", "country": "Mexico", "iata": "CUN"}

        result1 = chunk_sections(sections, meta)
        result2 = chunk_sections(sections, meta)

        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2):
            assert c1["id"] == c2["id"]

    def test_chunk_sections_skips_empty_sections(self):
        sections = [
            {"heading": "See", "level": "2", "content": ""},
            {"heading": "Do", "level": "2", "content": "   "},
            {"heading": "Eat", "level": "2", "content": "Local food."},
        ]
        meta = {"city": "Cancun", "country": "Mexico", "iata": "CUN"}
        result = chunk_sections(sections, meta)

        headings = [c["heading"] for c in result]
        assert "See" not in headings
        assert "Do" not in headings
        assert "Eat" in headings

    def test_chunk_sections_id_format(self):
        sections = [
            {"heading": "See", "level": "2", "content": "Visit the cathedral."},
        ]
        meta = {"city": "Barcelona", "country": "Spain", "iata": "BCN"}
        result = chunk_sections(sections, meta)

        assert len(result) >= 1
        for chunk in result:
            # ID format: {iata}_{hex_hash}
            assert chunk["id"].startswith("BCN_")
            # The hash part should be exactly 16 hex characters
            parts = chunk["id"].split("_", 1)
            assert len(parts) == 2
            assert re.match(r"^[0-9a-f]{16}$", parts[1])

    def test_chunk_sections_multiple_sections(self):
        sections = [
            {"heading": "See", "level": "2", "content": "See content."},
            {"heading": "Do", "level": "2", "content": "Do content."},
            {"heading": "Eat", "level": "2", "content": "Eat content."},
        ]
        meta = {"city": "Paris", "country": "France", "iata": "CDG"}
        result = chunk_sections(sections, meta)

        assert len(result) == 3
        # Each chunk should have its heading prepended
        for chunk in result:
            assert chunk["heading"] in chunk["text"]
