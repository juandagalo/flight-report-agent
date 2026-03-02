"""FastAPI application — Travel Recommendation Agent."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel

from src.app.config import settings
from src.app.graph.pipeline import travel_graph
from src.app.templates import get_graph_viewer_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger(__name__)

MAX_REPORT_AGE_SECONDS = 3600  # 1 hour


def _cleanup_old_reports() -> int:
    """Remove PDF reports older than MAX_REPORT_AGE_SECONDS. Returns count removed."""
    report_dir = settings.REPORT_OUTPUT_DIR
    if not os.path.isdir(report_dir):
        return 0

    now = time.time()
    removed = 0
    for name in os.listdir(report_dir):
        if not name.endswith(".pdf"):
            continue
        path = os.path.join(report_dir, name)
        try:
            if now - os.path.getmtime(path) > MAX_REPORT_AGE_SECONDS:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle: clean up stale PDF reports."""
    removed = _cleanup_old_reports()
    if removed:
        logger.info("Startup cleanup: removed %d stale report(s)", removed)
    yield
    removed = _cleanup_old_reports()
    if removed:
        logger.info("Shutdown cleanup: removed %d stale report(s)", removed)


app = FastAPI(
    title="Travel Recommendation Agent",
    description=(
        "Agente LangGraph que genera un informe comparativo de viaje en PDF "
        "basado en las preferencias del usuario y datos reales de vuelos (Amadeus)."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global exception handler ─────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a consistent JSON error for unhandled exceptions."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor",
            "error": str(exc),
        },
    )


# ── Endpoints ─────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Basic health-check."""
    return {"status": "ok"}


@app.get("/api/graph/ascii", response_class=PlainTextResponse)
async def get_graph_ascii():
    """Get the LangGraph pipeline as an ASCII diagram."""
    try:
        graph = travel_graph.get_graph()
        ascii_diagram = graph.draw_ascii()
        return ascii_diagram
    except Exception as exc:
        logger.error("Failed to generate ASCII diagram: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error generating diagram: {exc}")


@app.get("/api/graph/viewer", response_class=HTMLResponse)
async def get_graph_viewer():
    """Interactive Mermaid diagram viewer."""
    try:
        graph = travel_graph.get_graph()
        mermaid_diagram = graph.draw_mermaid()
        return get_graph_viewer_html(mermaid_diagram)
    except Exception as exc:
        logger.error("Failed to generate graph viewer: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error generating viewer: {exc}")


async def _create_travel_report(initial_state: dict):
    """Run the full LangGraph pipeline and return the PDF report (internal).

    The pipeline: intake → validate → suggest destinations → search flights →
    enrich (weather + activities) → generate PDF.
    """
    try:
        result = await travel_graph.ainvoke(initial_state)
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


# ── Chat endpoint (natural language) ─────────────────────────────────


class ChatMessage(BaseModel):
    """Request body for the chat endpoint."""
    message: str


@app.post("/api/chat")
async def chat(body: ChatMessage):
    """Accept a natural-language travel request, run the pipeline, and return the PDF.

    Example: ``{"message": "Quiero ir a la playa desde Bogotá en julio, presupuesto 1500 USD"}``
    """
    logger.info("New chat request: %s", body.message[:120])

    initial_state = {
        "user_message": body.message,
        "validated": False,
        "validation_errors": [],
        "candidate_destinations": [],
        "destination_reports": [],
        "enriched": False,
        "report_path": "",
        "suggest_retry_count": 0,
        "errors": [],
    }

    return await _create_travel_report(initial_state)
