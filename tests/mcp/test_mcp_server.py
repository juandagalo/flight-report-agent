"""Tests for the MCP server -- tools definition and call handling."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.app.mcp.server import call_tool, list_tools
from src.app.mcp.tools import FlightReportInput, FlightReportOutput
from src.app.schemas import (
    CandidateDestination,
    DestinationReport,
    FlightOffer,
    WeatherInfo,
)


def _make_report(
    city: str = "Cancun",
    iata: str = "CUN",
    country: str = "Mexico",
    score: int = 80,
    price: float = 750.0,
) -> DestinationReport:
    return DestinationReport(
        destination=CandidateDestination(
            city=city,
            iata_code=iata,
            country=country,
            reasoning="Great destination",
            climate_match=85,
            activity_match=70,
        ),
        flights=[
            FlightOffer(
                airline="TestAir",
                price=price,
                currency="USD",
                departure="2026-07-01T08:00:00",
                arrival="2026-07-01T12:00:00",
                duration="PT4H",
                stops=0,
            ),
        ],
        weather=WeatherInfo(
            avg_temp_c=28.0,
            min_temp_c=24.0,
            max_temp_c=32.0,
            avg_precipitation_mm=2.0,
            description="Sunny and warm",
        ),
        activities_description="Snorkeling, beach walks",
        overall_score=score,
    )


class TestListTools:
    async def test_list_tools_returns_flight_report(self):
        tools = await list_tools()

        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "flight_report"
        assert "travel report" in tool.description.lower()
        assert tool.inputSchema is not None

    async def test_list_tools_schema_has_required_fields(self):
        tools = await list_tools()
        schema = tools[0].inputSchema

        assert "properties" in schema
        assert "message" in schema["properties"]
        assert "user_id" in schema["properties"]
        assert "message" in schema.get("required", [])


class TestCallToolUnknown:
    async def test_call_tool_unknown_name(self):
        result = await call_tool("unknown_tool", {})

        assert len(result) == 1
        assert "Unknown tool" in result[0].text

    async def test_call_tool_unknown_name_contains_tool_name(self):
        result = await call_tool("nonexistent", {})

        assert "nonexistent" in result[0].text


class TestCallToolValidation:
    async def test_call_tool_invalid_input_missing_message(self):
        result = await call_tool("flight_report", {})

        assert len(result) == 1
        assert "Invalid input" in result[0].text

    async def test_call_tool_invalid_input_int_coerced(self):
        """Pydantic coerces int to str, so this should succeed without crashing."""
        result = await call_tool("flight_report", {"message": 123})
        assert len(result) == 1

    async def test_call_tool_invalid_input_none_value(self):
        """Passing None for message -- should either coerce or return a validation error."""
        result = await call_tool("flight_report", {"message": None})
        assert len(result) == 1


class TestCallToolSuccess:
    @patch("src.app.mcp.server.travel_graph")
    async def test_call_tool_success(self, mock_graph):
        reports = [
            _make_report("Cancun", "CUN", "Mexico", 85, 750.0),
            _make_report("Punta Cana", "PUJ", "Dominican Republic", 78, 900.0),
        ]
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "report_path": "/tmp/report.pdf",
                "destination_reports": reports,
                "errors": [],
            }
        )

        result = await call_tool(
            "flight_report",
            {"message": "Quiero ir a la playa desde Bogota en julio"},
        )

        assert len(result) == 1
        output = json.loads(result[0].text)
        assert output["success"] is True
        assert output["report_path"] == "/tmp/report.pdf"
        assert len(output["destinations"]) == 2
        assert "CUN" in output["destinations"][0]
        assert "PUJ" in output["destinations"][1]
        assert output["summary"] != ""
        assert output["errors"] == []

    @patch("src.app.mcp.server.travel_graph")
    async def test_call_tool_success_with_user_id(self, mock_graph):
        reports = [_make_report()]
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "report_path": "/tmp/report.pdf",
                "destination_reports": reports,
                "errors": [],
            }
        )

        result = await call_tool(
            "flight_report",
            {"message": "Beach vacation", "user_id": "user-42"},
        )

        # Verify user_id was passed in the initial state
        call_args = mock_graph.ainvoke.call_args[0][0]
        assert call_args["user_id"] == "user-42"

        output = json.loads(result[0].text)
        assert output["success"] is True

    @patch("src.app.mcp.server.travel_graph")
    async def test_call_tool_summary_sorted_by_score(self, mock_graph):
        reports = [
            _make_report("Cancun", "CUN", "Mexico", 70, 800.0),
            _make_report("Punta Cana", "PUJ", "Dominican Republic", 90, 1200.0),
        ]
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "report_path": "/tmp/report.pdf",
                "destination_reports": reports,
                "errors": [],
            }
        )

        result = await call_tool(
            "flight_report",
            {"message": "Beach vacation"},
        )

        output = json.loads(result[0].text)
        summary_lines = output["summary"].split("\n")
        # Punta Cana (score 90) should appear before Cancun (score 70)
        assert "Punta Cana" in summary_lines[0]
        assert "Cancun" in summary_lines[1]


class TestCallToolPipelineFailure:
    @patch("src.app.mcp.server.travel_graph")
    async def test_call_tool_pipeline_failure(self, mock_graph):
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Pipeline exploded"))

        result = await call_tool(
            "flight_report",
            {"message": "Beach vacation from Bogota"},
        )

        assert len(result) == 1
        output = json.loads(result[0].text)
        assert output["success"] is False
        assert len(output["errors"]) > 0
        assert "Pipeline exploded" in output["errors"][0]


class TestCallToolNoDestinations:
    @patch("src.app.mcp.server.travel_graph")
    async def test_call_tool_no_destinations(self, mock_graph):
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "report_path": "",
                "destination_reports": [],
                "errors": [],
            }
        )

        result = await call_tool(
            "flight_report",
            {"message": "Beach vacation"},
        )

        output = json.loads(result[0].text)
        assert output["success"] is False
        assert output["destinations"] == []
        assert output["summary"] == "No destinations found."
        assert output["report_path"] == ""

    @patch("src.app.mcp.server.travel_graph")
    async def test_call_tool_errors_in_result(self, mock_graph):
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "report_path": "",
                "destination_reports": [],
                "errors": ["No flights found for any destination"],
            }
        )

        result = await call_tool(
            "flight_report",
            {"message": "Beach vacation"},
        )

        output = json.loads(result[0].text)
        assert output["success"] is False
        assert "No flights found" in output["errors"][0]


class TestCallToolInitialState:
    @patch("src.app.mcp.server.travel_graph")
    async def test_initial_state_matches_main_pattern(self, mock_graph):
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "report_path": "",
                "destination_reports": [],
                "errors": [],
            }
        )

        await call_tool(
            "flight_report",
            {"message": "Test message", "user_id": "test-user"},
        )

        call_args = mock_graph.ainvoke.call_args[0][0]
        assert call_args["user_message"] == "Test message"
        assert call_args["user_id"] == "test-user"
        assert call_args["validated"] is False
        assert call_args["validation_errors"] == []
        assert call_args["candidate_destinations"] == []
        assert call_args["destination_reports"] == []
        assert call_args["enriched"] is False
        assert call_args["report_path"] == ""
        assert call_args["suggest_retry_count"] == 0
        assert call_args["errors"] == []


class TestFlightReportInputSchema:
    def test_schema_has_expected_fields(self):
        schema = FlightReportInput.model_json_schema()

        assert "properties" in schema
        props = schema["properties"]
        assert "message" in props
        assert "user_id" in props
        assert "message" in schema.get("required", [])

    def test_message_is_required(self):
        with pytest.raises(Exception):
            FlightReportInput()

    def test_user_id_defaults_to_empty(self):
        inp = FlightReportInput(message="test")
        assert inp.user_id == ""

    def test_valid_input(self):
        inp = FlightReportInput(
            message="Quiero ir a la playa",
            user_id="user-1",
        )
        assert inp.message == "Quiero ir a la playa"
        assert inp.user_id == "user-1"


class TestFlightReportOutputSchema:
    def test_output_defaults(self):
        out = FlightReportOutput(success=True)
        assert out.report_path == ""
        assert out.destinations == []
        assert out.errors == []
        assert out.summary == ""

    def test_output_serialization(self):
        out = FlightReportOutput(
            success=True,
            report_path="/tmp/report.pdf",
            destinations=["Cancun (CUN) - score: 85"],
            summary="- Cancun, Mexico (score: 85, from $750)",
        )
        data = json.loads(out.model_dump_json())
        assert data["success"] is True
        assert data["report_path"] == "/tmp/report.pdf"
        assert len(data["destinations"]) == 1
