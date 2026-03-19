"""MCP tool definitions for the flight report agent."""

from pydantic import BaseModel, Field


class FlightReportInput(BaseModel):
    """Input schema for the flight_report tool."""

    message: str = Field(
        ...,
        description=(
            "Natural language travel request in Spanish or English. "
            "Example: 'Quiero ir a la playa desde Bogota en julio, "
            "presupuesto 1500 USD para 2 personas'"
        ),
    )
    user_id: str = Field(
        default="",
        description="Optional user identifier for personalized recommendations.",
    )


class FlightReportOutput(BaseModel):
    """Output schema for the flight_report tool."""

    success: bool
    report_path: str = Field(default="", description="Path to generated PDF report")
    destinations: list[str] = Field(
        default_factory=list,
        description="List of recommended destinations with scores",
    )
    errors: list[str] = Field(default_factory=list)
    summary: str = Field(default="", description="Brief text summary of recommendations")
