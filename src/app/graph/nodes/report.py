"""Node: Generate the PDF report."""

from __future__ import annotations

import logging

from src.app.config import settings
from src.app.graph.state import TravelState
from src.app.services.pdf_generator import generate_report

logger = logging.getLogger(__name__)


def generate_report_node(state: TravelState) -> TravelState:
    """Build the comparative PDF report."""

    logger.info("→ Starting GENERATE_REPORT node")

    reports = state.get("destination_reports", [])
    request = state.get("request")
    if request is None:
        return {**state, "report_path": "", "errors": state.get("errors", []) + ["No request found in state"]}

    if not reports:
        return {
            **state,
            "report_path": "",
            "errors": state.get("errors", []) + [
                "No hay destinos con vuelos disponibles para generar el informe."
            ],
        }

    try:
        path = generate_report(
            request=request,
            destination_reports=reports,
            output_dir=settings.REPORT_OUTPUT_DIR,
        )

        logger.info("Report generated at %s", path)
        
        return {
            **state,
            "report_path": path,
        }
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        return {
            **state,
            "report_path": "",
            "errors": state.get("errors", []) + [
                f"Error al generar el PDF: {exc}"
            ],
        }
