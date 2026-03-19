# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the dev server
uv run uvicorn src.app.main:app --reload

# Run tests
uv run pytest -v

# Ingest WikiVoyage knowledge base (one-time, requires OPENAI_API_KEY)
uv run ingest-wikivoyage

# Run MCP server (stdio)
uv run flight-report-mcp
```

No linter or formatter is configured.

### Tests

195 tests across 22 files using **pytest** + **pytest-asyncio** (async auto mode). All external I/O (LLM, Amadeus, Open-Meteo, PDF, embeddings) is mocked; pure logic and Qdrant embedded mode (with `tmp_path`) are tested directly.

```
tests/
  conftest.py                          # env-var setup, shared fixtures
  test_schemas.py                      # Pydantic validation constraints
  graph/
    test_pipeline.py                   # Router functions, retry logic, errors reducer
    nodes/
      test_validate.py                 # Pure logic -- no mocks
      test_intake.py                   # Mock get_llm
      test_suggest.py                  # Mock get_llm
      test_suggest_with_rag.py         # RAG context injection, fallback, graceful degradation
      test_search_flights.py           # Mock amadeus_client.search_flights
      test_enrich.py                   # _compute_score (pure) + mock weather/LLM
      test_enrich_with_rag.py          # RAG context injection, fallback, graceful degradation
      test_report.py                   # Mock pdf_generator.generate_report
      test_interaction_storage.py      # Storage with/without user_id, non-fatal errors
  services/
    test_amadeus_client.py             # _parse_offers (pure) + SDK mock
    test_weather_client.py             # Mock httpx.AsyncClient
    test_pdf_generator.py              # Format helpers (pure) + real PDF to tmp_path
    test_llm_provider.py               # Factory: both providers, unknown, case, whitespace
    test_embeddings.py                 # Singleton, provider independence, embed calls
    test_qdrant_client.py              # Real Qdrant embedded with tmp_path isolation
    test_rag.py                        # Query helpers, formatting, sanitization
    test_wikivoyage.py                 # Fetch, parse, strip wikitext (all HTTP mocked)
    test_chunker.py                    # Token counting, splitting, overlap, metadata
  mcp/
    test_mcp_server.py                 # Tool listing, invocation, validation, error handling
  integration/
    test_full_pipeline_rag.py          # E2E with real Qdrant, mocked external APIs
  app/
    test_main.py                       # FastAPI endpoints + SSE streaming + PDF download
```

## Architecture

This is a **FastAPI + LangGraph** agent that generates comparative PDF travel reports. Users submit travel preferences via natural language (`POST /api/chat`) and receive a downloadable PDF. An SSE streaming endpoint (`POST /api/chat/stream`) provides real-time node-by-node progress events.

### LLM Provider

The agent supports multiple LLM providers via `LLM_PROVIDER` env var:
- `claude` (default) -- uses `ChatAnthropic` via `langchain-anthropic`
- `openai` -- uses `ChatOpenAI` via `langchain-openai`

All nodes call `get_llm(temperature)` from `src/app/services/llm_provider.py` instead of instantiating a provider directly. Embeddings always use OpenAI `text-embedding-3-small` regardless of LLM provider (Anthropic has no embedding API).

### LangGraph Pipeline (`src/app/graph/pipeline.py`)

```
intake -> validate -> suggest -> search_flights ->[retry?]-> enrich -> generate_report -> store_interaction -> END
                        ^             |
                        +-------------+  (retry via increment_retry if no affordable flights, max 1)
```

- **State** flows as `TravelState` (TypedDict in `src/app/graph/state.py`) through all nodes
- All nodes are `async def` and return **delta dicts** (only changed keys), not full state copies
- `suggest` and `enrich` nodes query Qdrant for RAG context before LLM calls, with graceful fallback
- `store_interaction` saves completed requests to Qdrant for future personalization (non-fatal on errors)
- Conditional edges handle validation routing and flight search retry logic

### RAG Layer (Qdrant)

Two Qdrant collections in embedded mode (no external server):

| Collection | Purpose | Populated by |
|---|---|---|
| `travel_knowledge` | WikiVoyage destination guides (chunked, embedded) | `uv run ingest-wikivoyage` |
| `interactions` | Past user requests + results for personalization | `store_interaction` pipeline node |

RAG context is injected into `suggest` and `enrich` prompts via `src/app/services/rag.py`. Includes prompt injection sanitization (`_sanitize_rag_text`) for WikiVoyage content.

### Key Layers

| Layer | Location | Purpose |
|---|---|---|
| API | `src/app/main.py` | FastAPI routes (`/health`, `/api/chat`, `/api/chat/stream`, `/api/reports/{filename}`, graph visualization) |
| Schemas | `src/app/schemas.py` | Pydantic models: `TravelRequest`, `IntakeResult`, `CandidateDestination`, `FlightOffer`, `WeatherInfo`, `DestinationReport` |
| Graph nodes | `src/app/graph/nodes/` | One file per pipeline step: `intake`, `validate`, `suggest`, `search_flights`, `enrich`, `report`, `store_interaction` |
| Services | `src/app/services/` | `llm_provider` (factory), `embeddings` (OpenAI), `qdrant_client` (vector store), `rag` (query helpers), `amadeus_client` (flights), `weather_client` (Open-Meteo), `pdf_generator` (ReportLab) |
| Scraper | `src/app/services/scraper/` | `wikivoyage` (MediaWiki API), `chunker` (token-aware splitting), `destinations` (curated list), `ingest` (CLI pipeline) |
| Prompts | `src/app/prompts/templates.py` | LLM prompt templates (Spanish), including RAG-enhanced variants with defensive instructions |
| Config | `src/app/config.py` | `pydantic-settings` singleton loading from `.env` |
| MCP | `src/app/mcp/` | MCP server (`server.py`) and tool schemas (`tools.py`) for Claude Code integration |
| Skill | `skills/flight-report/SKILL.md` | Claude Code skill definition with trigger patterns |

### External APIs

- **Amadeus** (requires API keys) -- flight search via `amadeus` Python SDK; supports test/production environments; retries on 5xx errors
- **Open-Meteo** (free, no key) -- historical weather data as climate proxy; uses hardcoded city coordinates
- **OpenAI** -- LLM (when `LLM_PROVIDER=openai`) and embeddings (always, `text-embedding-3-small`)
- **Anthropic Claude** -- LLM (when `LLM_PROVIDER=claude`)
- **WikiVoyage** -- MediaWiki API for destination knowledge (scraped during ingestion, not at runtime)

### Conventions

- All LLM prompts and user-facing PDF content are in **Spanish**
- The `intake` node converts natural-language messages to `TravelRequest` via structured output; passes through when `user_message` is empty
- The `suggest` node uses LangChain's `with_structured_output` (JSON mode) to get typed `DestinationList` from the LLM
- Weather fetching is async (`httpx.AsyncClient`), gathered concurrently via `asyncio.gather`
- Sync operations (Amadeus SDK, PDF generation) are wrapped with `asyncio.to_thread`
- PDFs are written to `reports/` directory (configurable via `REPORT_OUTPUT_DIR`); stale reports (>1 hour) are cleaned on app startup/shutdown
- Budget filtering happens in `search_flights_node`: compares per-offer price against `max_budget * num_people`
- RAG calls in pipeline nodes are wrapped in `try/except` -- failures log warnings and fall back to non-RAG templates
- Qdrant uses embedded mode with disk persistence at `QDRANT_PATH` (default: `data/qdrant`)
- CORS is enabled (allow all origins for dev)
- Global exception handler returns consistent JSON error shape

### Required Environment Variables

```
OPENAI_API_KEY          # Always required (embeddings use OpenAI)
AMADEUS_CLIENT_ID
AMADEUS_CLIENT_SECRET
```

Optional:
```
LLM_PROVIDER=claude                         # "claude" (default) or "openai"
ANTHROPIC_API_KEY=                          # Required when LLM_PROVIDER=claude
ANTHROPIC_MODEL=claude-sonnet-4-20250514  # Claude model
OPENAI_MODEL=gpt-4o                         # OpenAI model
AMADEUS_ENV=test                            # "test" or "production"
QDRANT_PATH=data/qdrant                     # Qdrant embedded storage path
REPORT_OUTPUT_DIR=reports                   # PDF output directory
```
