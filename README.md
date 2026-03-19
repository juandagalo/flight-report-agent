# Travel Recommendation Agent

A FastAPI + LangGraph agent that generates a **comparative PDF travel report** based on user preferences. Given an origin airport, budget, travel dates, and preferred activities, the agent suggests destinations, searches for real flights via the Amadeus API, enriches the results with weather data, and produces a downloadable PDF report.

Users interact via **natural-language messages** through the chat endpoint, which returns the PDF directly. The agent also exposes an **MCP server** for integration with Claude Code and other MCP clients.

## v2 Features

- **Multi-provider LLM support** -- Switch between OpenAI and Anthropic Claude via the `LLM_PROVIDER` environment variable. All LLM-using nodes (intake, suggest, enrich) go through a provider-agnostic factory function.
- **RAG-enhanced recommendations** -- A Qdrant vector store (embedded mode, no external server) provides destination knowledge from WikiVoyage and user interaction history. The suggest and enrich nodes incorporate RAG context into their prompts when available, with graceful fallback to original prompts when Qdrant is empty or unavailable.
- **WikiVoyage knowledge base ingestion** -- A scraper + chunker + ingestion pipeline fetches destination content from WikiVoyage, splits it into token-aware chunks, embeds them via OpenAI `text-embedding-3-small`, and stores them in the `travel_knowledge` Qdrant collection.
- **User interaction history** -- After each pipeline run, a summary of the user's request and results is stored in the `interactions` Qdrant collection (when a `user_id` is provided). Future requests from the same user benefit from personalized context.
- **MCP server** -- The agent is exposed as an MCP tool (`flight_report`) for integration with Claude Desktop, Claude Code, and other MCP-compatible clients.
- **Claude Code skill** -- A skill definition at `skills/flight-report/SKILL.md` enables Claude Code to automatically invoke the flight report tool in response to travel-related requests.

## Architecture

```
intake -> validate -> suggest -> search_flights -> enrich -> generate_report -> store_interaction -> END
                        |                            |                                |
                        v                            v                                v
                  [Qdrant RAG]                 [Qdrant RAG]                   [Qdrant Store]
               travel_knowledge             travel_knowledge                  interactions
                  interactions
                        ^__________________________|
                          (retry if no flights found)
```

1. **intake** -- extracts structured travel preferences from natural language (passthrough for JSON requests)
2. **validate** -- checks date ordering and missing request (field constraints enforced by Pydantic schemas)
3. **suggest** -- LLM proposes candidate destinations; queries Qdrant for destination knowledge and user history to enhance recommendations
4. **search_flights** -- queries Amadeus API for real flight offers; retries with new destinations if none found
5. **enrich** -- fetches weather data (Open-Meteo) and activity descriptions (LLM); queries Qdrant for destination-specific context
6. **generate_report** -- builds a comparative PDF report with scores, flights, weather, and activities
7. **store_interaction** -- persists a summary of the interaction to Qdrant for future personalization (requires `user_id`)

## Tech Stack

- **LangGraph** -- agent pipeline orchestration
- **FastAPI** -- REST API with SSE streaming
- **Amadeus API** -- real flight search
- **OpenAI / Anthropic Claude** -- destination suggestion, enrichment, and natural-language intake (configurable via `LLM_PROVIDER`)
- **Qdrant** -- vector store for RAG (embedded mode, disk-persisted)
- **OpenAI Embeddings** -- `text-embedding-3-small` (1536 dimensions) for all vector operations
- **ReportLab** -- PDF generation
- **MCP SDK** -- Model Context Protocol server for tool exposure

## Setup

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env  # fill in your keys

# Ingest WikiVoyage knowledge base (one-time, requires OPENAI_API_KEY)
uv run ingest-wikivoyage

# Run the FastAPI server
uvicorn src.app.main:app --reload

# Or run the MCP server (for Claude Code / Claude Desktop)
uv run flight-report-mcp
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | -- | OpenAI API key (used for LLM when provider is openai, and always for embeddings) |
| `AMADEUS_CLIENT_ID` | Yes | -- | Amadeus API client ID |
| `AMADEUS_CLIENT_SECRET` | Yes | -- | Amadeus API client secret |
| `AMADEUS_ENV` | No | `test` | Amadeus environment: `test` or `production` |
| `LLM_PROVIDER` | No | `openai` | LLM provider: `openai` or `claude` |
| `OPENAI_MODEL` | No | `gpt-4o` | OpenAI model name |
| `ANTHROPIC_API_KEY` | If claude | -- | Anthropic API key (required when `LLM_PROVIDER=claude`) |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-20250514` | Anthropic model name |
| `QDRANT_PATH` | No | `data/qdrant` | Qdrant embedded storage path (local disk) |
| `QDRANT_COLLECTION_KNOWLEDGE` | No | `travel_knowledge` | Qdrant collection for WikiVoyage content |
| `QDRANT_COLLECTION_INTERACTIONS` | No | `interactions` | Qdrant collection for user interaction history |
| `EMBEDDING_DIMENSION` | No | `1536` | Embedding vector dimension (matches text-embedding-3-small) |
| `REPORT_OUTPUT_DIR` | No | `reports` | Directory for generated PDF reports |

**Note:** `OPENAI_API_KEY` is always required, even when using Claude as the LLM provider, because embeddings use OpenAI's `text-embedding-3-small` model (Anthropic does not offer a standalone embedding API).

### MCP Server Configuration

To use the flight report agent from Claude Code or Claude Desktop, add this to your MCP settings:

```json
{
  "mcpServers": {
    "flight-report-agent": {
      "command": "uv",
      "args": ["--directory", "/path/to/flight-report-agent", "run", "flight-report-mcp"],
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

## Testing

```bash
uv run pytest -v
```

195 tests covering schemas, all graph nodes, services, pipeline logic, API endpoints, MCP server, RAG integration, and end-to-end integration. All external I/O is mocked; integration tests use real Qdrant in embedded mode with temporary directories.

## Running

```bash
uvicorn src.app.main:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `/docs`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/chat` | Natural-language travel request, returns PDF |
| `POST` | `/api/chat/stream` | SSE streaming version -- emits node-by-node progress events |
| `GET` | `/api/reports/{filename}` | Download a generated PDF report |
| `GET` | `/api/graph/ascii` | ASCII diagram of the LangGraph pipeline |
| `GET` | `/api/graph/viewer` | Interactive Mermaid graph viewer |

## Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quiero ir a la playa desde Bogota en julio, presupuesto 1500 USD para 2 personas"}' \
  --output informe.pdf
```

On success the response is the PDF file directly (`application/pdf`).

### With User ID (for personalized recommendations)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quiero ir a la playa desde Bogota en julio", "user_id": "juan-123"}' \
  --output informe.pdf
```

### SSE Streaming

For real-time progress updates, use the streaming endpoint:

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Quiero ir a la playa desde Bogota en julio, presupuesto 1500 USD para 2 personas"}'
```

Sample events:

```
event: node_start
data: {"node": "intake", "message": "Interpretando tu solicitud..."}

event: node_end
data: {"node": "intake", "message": "Solicitud interpretada"}

event: node_start
data: {"node": "validate", "message": "Validando datos de viaje..."}

...

event: node_end
data: {"node": "store_interaction", "message": "Preferencias guardadas"}

event: complete
data: {"report_url": "/api/reports/report_abc123.pdf", "message": "Informe PDF listo para descargar"}
```

Then download the PDF:

```bash
curl http://localhost:8000/api/reports/report_abc123.pdf --output informe.pdf
```
