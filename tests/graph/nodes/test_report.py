"""Tests for the generate_report node — mocks pdf_generator.generate_report."""

from __future__ import annotations

from unittest.mock import patch

from src.app.graph.nodes.report import generate_report_node


class TestGenerateReportNode:
    async def test_no_request(self):
        result = await generate_report_node({})
        assert result["report_path"] == ""
        assert any("No request" in e for e in result["errors"])

    async def test_no_reports(self, sample_travel_request):
        state = {"request": sample_travel_request, "destination_reports": []}
        result = await generate_report_node(state)
        assert result["report_path"] == ""
        assert any("No hay destinos" in e for e in result["errors"])

    @patch("src.app.graph.nodes.report.generate_report")
    async def test_successful_generation(self, mock_gen, sample_travel_request, sample_destination_report):
        mock_gen.return_value = "/tmp/report.pdf"

        state = {
            "request": sample_travel_request,
            "destination_reports": [sample_destination_report],
        }
        result = await generate_report_node(state)
        assert result["report_path"] == "/tmp/report.pdf"
        assert "errors" not in result

    @patch("src.app.graph.nodes.report.generate_report")
    async def test_pdf_failure_returns_error(self, mock_gen, sample_travel_request, sample_destination_report):
        mock_gen.side_effect = RuntimeError("ReportLab crash")

        state = {
            "request": sample_travel_request,
            "destination_reports": [sample_destination_report],
        }
        result = await generate_report_node(state)
        assert result["report_path"] == ""
        assert any("Error al generar" in e for e in result["errors"])
