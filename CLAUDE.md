# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the dev server
uvicorn src.app.main:app --reload

# Alternative entry point (runs on 0.0.0.0:8000 with reload)
python main.py

# Run tests
uv run pytest -v
```

No linter or formatter is configured.

### Tests

96 tests across 13 files using **pytest** + **pytest-asyncio** (async auto mode). All external I/O (OpenAI, Amadeus, Open-Meteo, PDF) is mocked; pure logic (`validate_input`, `_compute_score`, `_parse_offers`, format helpers) is tested directly.

```
tests/
  conftest.py                     # env-var setup, shared fixtures
  test_schemas.py                 # Pydantic validation constraints
  graph/
    test_pipeline.py              # Router functions, retry logic, errors reducer
    nodes/
      test_validate.py            # Pure logic — no mocks
      test_intake.py              # Mock ChatOpenAI
      test_suggest.py             # Mock ChatOpenAI
      test_search_flights.py      # Mock amadeus_client.search_flights
      test_enrich.py              # _compute_score (pure) + mock weather/LLM
      test_report.py              # Mock pdf_generator.generate_report
  services/
    test_amadeus_client.py        # _parse_offers (pure) + SDK mock
    test_weather_client.py        # Mock httpx.AsyncClient
    test_pdf_generator.py         # Format helpers (pure) + real PDF to tmp_path
  app/
    test_main.py                  # FastAPI endpoints + SSE streaming + PDF download
```

## Architecture

This is a **FastAPI + LangGraph** agent that generates comparative PDF travel reports. Users submit travel preferences via natural language (`POST /api/chat`) and receive a downloadable PDF. An SSE streaming endpoint (`POST /api/chat/stream`) provides real-time node-by-node progress events.

### LangGraph Pipeline (`src/app/graph/pipeline.py`)

```
intake → validate → suggest → search_flights →[retry?]→ enrich → generate_report → END
                       ↑            │
                       └────────────┘  (retry via increment_retry if no affordable flights found, max 1 retry)
```

- **State** flows as `TravelState` (TypedDict in `src/app/graph/state.py`) through all nodes
- All nodes are `async def` and return **delta dicts** (only changed keys), not full state copies
- The API uses `await travel_graph.ainvoke(...)` for proper async execution
- Conditional edges handle validation routing and flight search retry logic

### Key Layers

| Layer | Location | Purpose |
|-------|----------|---------|
| API | `src/app/main.py` | FastAPI routes (`/health`, `/api/chat`, `/api/chat/stream`, `/api/reports/{filename}`, graph visualization) |
| Schemas | `src/app/schemas.py` | Pydantic models: `TravelRequest`, `IntakeResult`, `CandidateDestination`, `FlightOffer`, `WeatherInfo`, `DestinationReport` |
| Graph nodes | `src/app/graph/nodes/` | One file per pipeline step: `intake`, `validate`, `suggest`, `search_flights`, `enrich`, `report` |
| Services | `src/app/services/` | External API wrappers: `amadeus_client` (flight search, cached singleton, retry on 5xx), `weather_client` (Open-Meteo, async), `pdf_generator` (ReportLab) |
| Prompts | `src/app/prompts/templates.py` | All LLM prompt templates (in Spanish), including intake extraction prompts |
| Config | `src/app/config.py` | `pydantic-settings` singleton loading from `.env`, warns on empty Amadeus credentials |

### External APIs

- **Amadeus** (requires API keys) — flight search via `amadeus` Python SDK; supports test/production environments; retries on 5xx errors; carrier names resolved from response-level `dictionaries`
- **Open-Meteo** (free, no key) — historical weather data as climate proxy; uses hardcoded city coordinates in `weather_client.py`
- **OpenAI GPT-4o** — intake extraction (natural language → structured request), destination suggestion (structured output / JSON mode), and activity enrichment

### Conventions

- All LLM prompts and user-facing PDF content are in **Spanish**
- The `intake` node converts natural-language messages to `TravelRequest` via structured output; passes through when `user_message` is empty
- The `suggest` node uses LangChain's `with_structured_output` (JSON mode) to get typed `DestinationList` from the LLM
- Weather fetching is async (`httpx.AsyncClient`), gathered concurrently via `asyncio.gather`
- Sync operations (Amadeus SDK, PDF generation) are wrapped with `asyncio.to_thread`
- PDFs are written to `reports/` directory (configurable via `REPORT_OUTPUT_DIR` setting); stale reports (>1 hour) are cleaned on app startup/shutdown
- Budget filtering happens in `search_flights_node`: compares per-offer price against `max_budget * num_people`
- CORS is enabled (allow all origins for dev)
- Global exception handler returns consistent JSON error shape

### Required Environment Variables

```
OPENAI_API_KEY
AMADEUS_CLIENT_ID
AMADEUS_CLIENT_SECRET
```

Optional: `OPENAI_MODEL` (default: gpt-4o), `AMADEUS_ENV` (default: test), `REPORT_OUTPUT_DIR` (default: reports).
