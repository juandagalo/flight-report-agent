# Travel Recommendation Agent

A FastAPI + LangGraph agent that generates a **comparative PDF travel report** based on user preferences. Given an origin airport, budget, travel dates, and preferred activities, the agent suggests destinations, searches for real flights via the Amadeus API, enriches the results with weather data, and produces a downloadable PDF report.

## Pipeline

```
validate → suggest destinations → search flights → enrich (weather + activities) → generate PDF
               ↑__________________________|
                  (retry if no flights found)
```

1. **validate** — checks required fields and date ranges
2. **suggest** — LLM proposes candidate destinations matching climate and activity preferences
3. **search_flights** — queries Amadeus API for real flight offers
4. **enrich** — fetches weather data and activity descriptions per destination
5. **generate_report** — builds a PDF comparing all destinations

## Tech Stack

- **LangGraph** — agent pipeline orchestration
- **FastAPI** — REST API
- **Amadeus API** — real flight search
- **OpenAI GPT-4o** — destination suggestion and enrichment
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
```

## Running

```bash
uvicorn src.app.main:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `/docs`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/travel-report` | Run the full pipeline, returns PDF |
| `GET` | `/api/graph/ascii` | ASCII diagram of the LangGraph pipeline |
| `GET` | `/api/graph/viewer` | Interactive Mermaid graph viewer |

## Example Request

```json
POST /api/travel-report
{
  "origin": "BOG",
  "region": "Caribe",
  "preferred_climate": "tropical",
  "preferred_activities": ["playa", "cultura"],
  "available_dates": [{ "date_from": "2025-07-01", "date_to": "2025-07-15" }],
  "max_budget": 1500,
  "currency": "USD",
  "num_people": 2
}
```
