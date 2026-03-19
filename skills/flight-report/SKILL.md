---
name: flight-report
description: Generate comparative travel reports with flight options, weather, and activity recommendations
tools:
  - mcp: flight-report-agent
    tool: flight_report
---

# Flight Report Agent

Generate a comparative travel report PDF based on user travel preferences.

## When to Use

- User asks for travel recommendations or flight comparisons
- User wants a PDF report of travel options
- User mentions travel planning, vacation planning, or trip comparison
- User asks about flights, destinations, or travel budgets

## Trigger Patterns

- "I want to travel to..." / "Quiero viajar a..."
- "Find me flights from..." / "Busca vuelos desde..."
- "Plan a trip to..." / "Planea un viaje a..."
- "Compare destinations for..." / "Compara destinos para..."
- "Generate a travel report" / "Genera un informe de viaje"
- "What are good destinations for..." / "Cuales son buenos destinos para..."

## Usage

The tool accepts a natural language travel request. Include:
- Origin city or airport
- Preferred climate or region
- Travel dates
- Budget (per person)
- Number of travelers
- Preferred activities

### Example Invocations

```
User: "I want to go to the beach from Bogota in July, budget $1500 for 2 people"
-> Invoke flight_report with message: "Quiero ir a la playa desde Bogota en julio, presupuesto 1500 USD para 2 personas"

User: "Compare European destinations for a cultural trip in September"
-> Invoke flight_report with message: "Quiero hacer un viaje cultural a Europa en septiembre desde Madrid, presupuesto 2000 EUR"
```

## MCP Server Setup

The flight-report-agent MCP server must be configured in Claude Code's MCP settings.
Add to `~/.claude/settings.local.json` or the project's `.claude/settings.local.json`:

```json
{
  "mcpServers": {
    "flight-report-agent": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/flight-report-agent", "run", "flight-report-mcp"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "LLM_PROVIDER": "claude",
        "AMADEUS_CLIENT_ID": "${AMADEUS_CLIENT_ID}",
        "AMADEUS_CLIENT_SECRET": "${AMADEUS_CLIENT_SECRET}"
      }
    }
  }
}
```

### Skill Installation

To register this skill with Claude Code, create a symlink:

```bash
ln -s /path/to/flight-report-agent/skills/flight-report ~/.claude/skills/flight-report
```

## Output

The tool returns a JSON response with:
- `success`: whether the report was generated
- `report_path`: local path to the PDF report
- `destinations`: list of recommended destinations with scores
- `summary`: text summary of top recommendations
- `errors`: any errors encountered
