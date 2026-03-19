"""MCP server for the flight report agent."""

import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.app.graph.pipeline import travel_graph
from src.app.mcp.tools import FlightReportInput, FlightReportOutput

logger = logging.getLogger(__name__)

server = Server("flight-report-agent")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available tools."""
    return [
        Tool(
            name="flight_report",
            description=(
                "Generate a comparative travel report with flight options, weather, "
                "and activity recommendations. Accepts natural language travel requests. "
                "Returns a PDF report path and text summary of destinations."
            ),
            inputSchema=FlightReportInput.model_json_schema(),
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocations."""
    if name != "flight_report":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        input_data = FlightReportInput(**arguments)
    except Exception as exc:
        return [TextContent(type="text", text=f"Invalid input: {exc}")]

    # Use pre-compiled graph singleton
    initial_state = {
        "user_message": input_data.message,
        "user_id": input_data.user_id,
        "validated": False,
        "validation_errors": [],
        "candidate_destinations": [],
        "destination_reports": [],
        "enriched": False,
        "report_path": "",
        "suggest_retry_count": 0,
        "errors": [],
    }

    try:
        result = await travel_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Pipeline failed in MCP call")
        output = FlightReportOutput(success=False, errors=[str(exc)])
        return [TextContent(type="text", text=output.model_dump_json(indent=2))]

    # Build output
    reports = result.get("destination_reports", [])
    destinations = [
        f"{r.destination.city} ({r.destination.iata_code}) - score: {r.overall_score}"
        for r in reports
    ]

    summary_parts = []
    for r in sorted(reports, key=lambda x: x.overall_score, reverse=True):
        cheapest = min((f.price for f in r.flights), default=0)
        summary_parts.append(
            f"- {r.destination.city}, {r.destination.country} "
            f"(score: {r.overall_score}, from ${cheapest:.0f})"
        )

    output = FlightReportOutput(
        success=bool(result.get("report_path")),
        report_path=result.get("report_path", ""),
        destinations=destinations,
        errors=result.get("errors", []),
        summary="\n".join(summary_parts) if summary_parts else "No destinations found.",
    )

    return [TextContent(type="text", text=output.model_dump_json(indent=2))]


async def run_server():
    """Run the MCP server over stdio."""
    logging.basicConfig(level=logging.INFO)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """CLI entry point for the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
