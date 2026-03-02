"""Tests for pdf_generator — format helpers (pure) + real PDF to tmp_path."""

from __future__ import annotations

import os

from src.app.services.pdf_generator import _format_duration, _format_datetime, generate_report


class TestFormatDuration:
    def test_standard(self):
        assert _format_duration("PT12H30M") == "12h 30m"

    def test_hours_only(self):
        assert _format_duration("PT5H") == "5h"

    def test_minutes_only(self):
        assert _format_duration("PT45M") == "45m"

    def test_empty(self):
        assert _format_duration("") == "—"


class TestFormatDatetime:
    def test_valid(self):
        result = _format_datetime("2026-07-01T08:30:00")
        assert result == "01/07/2026 08:30"

    def test_invalid(self):
        assert _format_datetime("not-a-date") == "not-a-date"

    def test_empty(self):
        assert _format_datetime("") == "—"


class TestGenerateReport:
    def test_creates_pdf(self, tmp_path, sample_travel_request, sample_destination_report):
        path = generate_report(
            request=sample_travel_request,
            destination_reports=[sample_destination_report],
            output_dir=str(tmp_path),
        )
        assert os.path.isfile(path)
        assert path.endswith(".pdf")
        assert os.path.getsize(path) > 0
