"""FastAPI application — Travel Recommendation Agent."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.app.config import settings
from src.app.graph.pipeline import travel_graph
from src.app.templates import get_graph_viewer_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger(__name__)

MAX_REPORT_AGE_SECONDS = 3600  # 1 hour

# ── Node-level progress messages (Spanish) ───────────────────────────

_NODE_MESSAGES: dict[str, tuple[str, str]] = {
    # node_name: (start_message, end_message)
    "intake":          ("Interpretando tu solicitud...",            "Solicitud interpretada"),
    "validate":        ("Validando datos de viaje...",             "Datos validados"),
    "suggest":         ("Sugiriendo destinos...",                  "Destinos sugeridos"),
    "search_flights":  ("Buscando vuelos en Amadeus...",           "Búsqueda de vuelos completada"),
    "increment_retry": ("Reintentando con nuevos destinos...",     "Reintento preparado"),
    "enrich":          ("Enriqueciendo con clima y actividades...", "Datos enriquecidos"),
    "generate_report":    ("Generando informe PDF...",                "Informe PDF generado"),
    "store_interaction":  ("Guardando preferencias del usuario...",   "Preferencias guardadas"),
}


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


def _build_initial_state(message: str, user_id: str = "") -> dict:
    """Build the initial TravelState dict from a user message."""
    return {
        "user_message": message,
        "user_id": user_id,
        "validated": False,
        "validation_errors": [],
        "candidate_destinations": [],
        "destination_reports": [],
        "enriched": False,
        "report_path": "",
        "suggest_retry_count": 0,
        "errors": [],
    }


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
    user_id: str = ""


@app.post("/api/chat")
async def chat(body: ChatMessage):
    """Accept a natural-language travel request, run the pipeline, and return the PDF.

    Example: ``{"message": "Quiero ir a la playa desde Bogotá en julio, presupuesto 1500 USD"}``
    """
    logger.info("New chat request: %s", body.message[:120])
    return await _create_travel_report(_build_initial_state(body.message, body.user_id))


# ── SSE streaming endpoint ───────────────────────────────────────────


def _sse_event(event: str, data: dict) -> dict:
    """Build a dict suitable for sse-starlette's EventSourceResponse."""
    return {"event": event, "data": json.dumps(data)}


async def _stream_travel_pipeline(
    initial_state: dict,
) -> AsyncGenerator[dict, None]:
    """Yield SSE events as the LangGraph pipeline progresses.

    Uses ``astream_events`` so that ``node_start`` is emitted *when a node
    begins* (not after it finishes), giving the client real-time progress.
    """
    try:
        async for event in travel_graph.astream_events(
            initial_state, version="v2"
        ):
            kind = event["event"]
            name = event.get("name", "")
            metadata = event.get("metadata", {})
            langgraph_node = metadata.get("langgraph_node", "")

            # Only handle top-level node events for known pipeline nodes
            if not langgraph_node or langgraph_node not in _NODE_MESSAGES:
                continue
            if name != langgraph_node:
                continue

            start_msg, end_msg = _NODE_MESSAGES[langgraph_node]

            if kind == "on_chain_start":
                yield _sse_event(
                    "node_start", {"node": langgraph_node, "message": start_msg}
                )

            elif kind == "on_chain_end":
                delta = event.get("data", {}).get("output", {})
                yield _sse_event(
                    "node_end", {"node": langgraph_node, "message": end_msg}
                )

                # Validation failure → emit error and stop
                if langgraph_node == "validate" and not delta.get("validated", False):
                    yield _sse_event(
                        "error",
                        {
                            "message": "Errores de validación",
                            "errors": delta.get("validation_errors", []),
                        },
                    )
                    return

                # Report generated → emit complete or error
                if langgraph_node == "generate_report":
                    report_path = delta.get("report_path", "")
                    if report_path and os.path.isfile(report_path):
                        filename = os.path.basename(report_path)
                        yield _sse_event(
                            "complete",
                            {
                                "report_url": f"/api/reports/{filename}",
                                "message": "Informe PDF listo para descargar",
                            },
                        )
                    else:
                        yield _sse_event(
                            "error",
                            {"message": "No se pudo generar el informe"},
                        )
                    return

    except Exception as exc:
        logger.exception("Streaming pipeline failed")
        yield _sse_event("error", {"message": f"Error en el pipeline: {exc}"})


@app.post("/api/chat/stream")
async def chat_stream(body: ChatMessage):
    """SSE streaming version of the chat endpoint.

    Emits node_start / node_end events as each pipeline step completes,
    then a final ``complete`` event with the report download URL.
    """
    logger.info("New streaming request: %s", body.message[:120])
    initial_state = _build_initial_state(body.message, body.user_id)
    return EventSourceResponse(_stream_travel_pipeline(initial_state))


# ── Report download endpoint ─────────────────────────────────────────


@app.get("/api/reports/{filename}")
async def download_report(filename: str):
    """Serve a generated PDF report by filename.

    Used by streaming clients to download the PDF after receiving the
    ``complete`` SSE event with a ``report_url``.
    """
    # Security: reject path traversal and non-PDF files
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo no válido")
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    path = os.path.join(settings.REPORT_OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Informe no encontrado")

    return FileResponse(
        path=path,
        media_type="application/pdf",
        filename=filename,
    )
