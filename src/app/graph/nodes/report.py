"""Node: Generate the PDF report."""

from __future__ import annotations

import asyncio
import logging

from src.app.config import settings
from src.app.graph.state import TravelState
from src.app.services.pdf_generator import generate_report

logger = logging.getLogger(__name__)


async def generate_report_node(state: TravelState) -> dict:
    """Build the comparative PDF report."""

    logger.info("→ Starting GENERATE_REPORT node")

    reports = state.get("destination_reports", [])
    request = state.get("request")
    if request is None:
        return {"report_path": "", "errors": ["No request found in state"]}

    if not reports:
        return {
            "report_path": "",
            "errors": [
                "No hay destinos con vuelos disponibles para generar el informe."
            ],
        }

    try:
        path = await asyncio.to_thread(
            generate_report,
            request=request,
            destination_reports=reports,
            output_dir=settings.REPORT_OUTPUT_DIR,
        )

        logger.info("Report generated at %s", path)

        return {
            "report_path": path,
        }
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        return {
            "report_path": "",
            "errors": [
                f"Error al generar el PDF: {exc}"
            ],
        }
