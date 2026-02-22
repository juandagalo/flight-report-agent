"""FastAPI application — Travel Recommendation Agent."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, HTMLResponse

from src.app.config import settings
from src.app.graph.pipeline import travel_graph
from src.app.schemas import TravelRequest
from src.app.templates import get_graph_viewer_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Travel Recommendation Agent",
    description=(
        "Agente LangGraph que genera un informe comparativo de viaje en PDF "
        "basado en las preferencias del usuario y datos reales de vuelos (Amadeus)."
    ),
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Basic health-check."""
    return {"status": "ok"}


@app.get("/api/graph/ascii", response_class=PlainTextResponse)
async def get_graph_ascii():
    """Get the LangGraph pipeline as an ASCII diagram.
    
    Returns a simple text representation of the graph structure.
    """
    try:
        graph = travel_graph.get_graph()
        ascii_diagram = graph.draw_ascii()
        return ascii_diagram
    except Exception as exc:
        logger.error("Failed to generate ASCII diagram: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error generating diagram: {exc}")


@app.get("/api/graph/viewer", response_class=HTMLResponse)
async def get_graph_viewer():
    """Interactive Mermaid diagram viewer.
    
    Returns an HTML page with the rendered graph visualization.
    """
    try:
        graph = travel_graph.get_graph()
        mermaid_diagram = graph.draw_mermaid()
        return get_graph_viewer_html(mermaid_diagram)
    except Exception as exc:
        logger.error("Failed to generate graph viewer: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error generating viewer: {exc}")


@app.post("/api/travel-report")
async def create_travel_report(request: TravelRequest):
    """Run the full LangGraph pipeline and return the PDF report.

    The pipeline: validate → suggest destinations → search flights →
    enrich (weather + activities) → generate PDF.
    """
    logger.info("New travel report request: origin=%s, region=%s", request.origin, request.region)

    initial_state = {
        "request": request,
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
        result = travel_graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Error en el pipeline: {exc}")

    # Check for validation errors
    if not result.get("validated", False):
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Errores de validación",
                "errors": result.get("validation_errors", []),
            },
        )

    # Check for report
    report_path = result.get("report_path", "")
    if not report_path or not os.path.isfile(report_path):
        errors = result.get("errors", [])
        return JSONResponse(
            status_code=500,
            content={
                "detail": "No se pudo generar el informe",
                "errors": errors,
            },
        )

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=os.path.basename(report_path),
    )
