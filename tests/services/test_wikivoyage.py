"""Tests for the WikiVoyage scraper -- mocks httpx.AsyncClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.app.services.scraper.wikivoyage import (
    fetch_page,
    parse_sections,
    strip_wikitext,
)


def _mock_success_response(title: str = "Cancun", wikitext: str = "== See ==\nBeaches.") -> MagicMock:
    """Build a mock httpx response for a successful WikiVoyage API call."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "parse": {
            "title": title,
            "wikitext": wikitext,
        }
    }
    return resp


def _mock_error_response() -> MagicMock:
    """Build a mock httpx response for a page-not-found error."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "error": {
            "code": "missingtitle",
            "info": "The page you specified doesn't exist.",
        }
    }
    return resp


class TestFetchPage:
    @patch("src.app.services.scraper.wikivoyage.httpx.AsyncClient")
    async def test_fetch_page_success(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_success_response("Cancun", "== See ==\nBeaches."))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await fetch_page("Cancun")
        assert result["title"] == "Cancun"
        assert result["wikitext"] == "== See ==\nBeaches."
        assert result["extract"] is None

    @patch("src.app.services.scraper.wikivoyage.httpx.AsyncClient")
    async def test_fetch_page_not_found(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_error_response())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await fetch_page("NonExistentPage")
        assert result["title"] == "NonExistentPage"
        assert result["wikitext"] is None

    @patch("src.app.services.scraper.wikivoyage.httpx.AsyncClient")
    async def test_fetch_page_http_error(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            await fetch_page("Cancun")

    @patch("src.app.services.scraper.wikivoyage.httpx.AsyncClient")
    async def test_fetch_page_wikitext_dict_format(self, mock_client_cls):
        """WikiVoyage API formatversion=1 returns wikitext as {"*": "..."}."""
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "parse": {
                "title": "Cancun",
                "wikitext": {"*": "== See ==\nBeaches and ruins."},
            }
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await fetch_page("Cancun")
        assert result["wikitext"] == "== See ==\nBeaches and ruins."


class TestParseSections:
    def test_parse_sections_basic(self):
        wikitext = "Lead paragraph.\n\n== See ==\nBeaches and ruins.\n\n== Do ==\nSnorkeling and diving."
        sections = parse_sections(wikitext)

        assert len(sections) == 3
        # Lead section
        assert sections[0]["heading"] == ""
        assert "Lead paragraph" in sections[0]["content"]
        # See section
        assert sections[1]["heading"] == "See"
        assert sections[1]["level"] == "2"
        assert "Beaches" in sections[1]["content"]
        # Do section
        assert sections[2]["heading"] == "Do"
        assert sections[2]["level"] == "2"
        assert "Snorkeling" in sections[2]["content"]

    def test_parse_sections_nested_headings(self):
        wikitext = "== See ==\nOverview.\n\n=== Beaches ===\nWhite sand.\n\n=== Ruins ===\nMayan ruins."
        sections = parse_sections(wikitext)

        headings = [s["heading"] for s in sections]
        levels = [s["level"] for s in sections]
        assert "See" in headings
        assert "Beaches" in headings
        assert "Ruins" in headings
        # The nested headings should have level 3
        beaches_idx = headings.index("Beaches")
        assert levels[beaches_idx] == "3"

    def test_parse_sections_empty_input(self):
        assert parse_sections("") == []
        assert parse_sections("   ") == []

    def test_parse_sections_no_headings(self):
        wikitext = "Just some plain text without any headings."
        sections = parse_sections(wikitext)
        # Should produce a lead section
        assert len(sections) == 1
        assert sections[0]["heading"] == ""
        assert "plain text" in sections[0]["content"]


class TestStripWikitext:
    def test_strip_wikitext_links(self):
        assert strip_wikitext("[[Paris|City of Light]]") == "City of Light"

    def test_strip_wikitext_simple_links(self):
        assert strip_wikitext("[[Paris]]") == "Paris"

    def test_strip_wikitext_templates(self):
        result = strip_wikitext("{{listing|name=Hotel|price=100}}")
        assert result == ""
        assert "listing" not in result

    def test_strip_wikitext_html(self):
        result = strip_wikitext("Hello<ref>citation</ref> world")
        assert "ref" not in result
        assert "citation" not in result

    def test_strip_wikitext_html_tags_removed(self):
        result = strip_wikitext("<div class='info'>Content</div>")
        assert "<div" not in result
        assert "</div>" not in result
        assert "Content" in result

    def test_strip_wikitext_bold_italic(self):
        result = strip_wikitext("'''bold''' and ''italic''")
        assert "bold" in result
        assert "italic" in result
        assert "'''" not in result

    def test_strip_wikitext_collapses_blank_lines(self):
        result = strip_wikitext("Line 1\n\n\n\n\nLine 2")
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_strip_wikitext_nested_templates(self):
        result = strip_wikitext("Before {{outer|{{inner}}}} after")
        assert "outer" not in result
        assert "inner" not in result
        assert "Before" in result
        assert "after" in result
