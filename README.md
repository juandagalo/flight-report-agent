# Travel Recommendation Agent

A FastAPI + LangGraph agent that generates a **comparative PDF travel report** based on user preferences. Given an origin airport, budget, travel dates, and preferred activities, the agent suggests destinations, searches for real flights via the Amadeus API, enriches the results with weather data, and produces a downloadable PDF report.

Users interact via **natural-language messages** through the chat endpoint, which returns the PDF directly.

## Pipeline

```
intake → validate → suggest destinations → search flights → enrich (weather + activities) → generate PDF
                          ↑__________________________|
                             (retry if no flights found)
```

1. **intake** — extracts structured travel preferences from natural language (passthrough for JSON requests)
2. **validate** — checks date ordering and missing request (field constraints enforced by Pydantic schemas)
3. **suggest** — LLM proposes candidate destinations matching climate and activity preferences
4. **search_flights** — queries Amadeus API for real flight offers
5. **enrich** — fetches weather data and activity descriptions per destination
6. **generate_report** — builds a PDF comparing all destinations

## Tech Stack

- **LangGraph** — agent pipeline orchestration
- **FastAPI** — REST API
- **Amadeus API** — real flight search
- **OpenAI GPT-4o** — destination suggestion, enrichment, and natural-language intake
- **ReportLab** — PDF generation

## Setup

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env  # fill in your keys
```

### Required environment variables

```env
OPENAI_API_KEY=sk-...
AMADEUS_CLIENT_ID=...
AMADEUS_CLIENT_SECRET=...
AMADEUS_ENV=test  # "test" or "production"
```

## Testing

```bash
uv run pytest -v
```

96 tests covering schemas, all graph nodes, services, pipeline logic, and API endpoints. All external I/O is mocked.

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
| `POST` | `/api/chat/stream` | SSE streaming version — emits node-by-node progress events |
| `GET` | `/api/reports/{filename}` | Download a generated PDF report |
| `GET` | `/api/graph/ascii` | ASCII diagram of the LangGraph pipeline |
| `GET` | `/api/graph/viewer` | Interactive Mermaid graph viewer |

## Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quiero ir a la playa desde Bogotá en julio, presupuesto 1500 USD para 2 personas"}' \
  --output informe.pdf
```

On success the response is the PDF file directly (`application/pdf`).

### SSE Streaming

For real-time progress updates, use the streaming endpoint:

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Quiero ir a la playa desde Bogotá en julio, presupuesto 1500 USD para 2 personas"}'
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

event: complete
data: {"report_url": "/api/reports/report_abc123.pdf", "message": "Informe PDF listo para descargar"}
```

Then download the PDF:

```bash
curl http://localhost:8000/api/reports/report_abc123.pdf --output informe.pdf
```
