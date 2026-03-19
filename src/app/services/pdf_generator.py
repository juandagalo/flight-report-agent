"""PDF report generator using ReportLab."""

from __future__ import annotations

import os
import re
import uuid
import logging
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)

from src.app.schemas import DestinationReport, TravelRequest

logger = logging.getLogger(__name__)


# ── Styles ───────────────────────────────────────────────────────────────

def _styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        "CoverTitle",
        parent=ss["Title"],
        fontSize=28,
        leading=34,
        spaceAfter=20,
        textColor=colors.HexColor("#1a3c5e"),
    ))
    ss.add(ParagraphStyle(
        "CoverSubtitle",
        parent=ss["Normal"],
        fontSize=14,
        leading=18,
        spaceAfter=8,
        textColor=colors.HexColor("#4a4a4a"),
    ))
    ss.add(ParagraphStyle(
        "SectionHeader",
        parent=ss["Heading1"],
        fontSize=18,
        leading=22,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.HexColor("#1a3c5e"),
    ))
    ss.add(ParagraphStyle(
        "SubHeader",
        parent=ss["Heading2"],
        fontSize=14,
        leading=17,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#2d6a9f"),
    ))
    ss.add(ParagraphStyle(
        "BodyText2",
        parent=ss["Normal"],
        fontSize=10,
        leading=13,
        spaceAfter=4,
    ))
    ss.add(ParagraphStyle(
        "ScoreHigh",
        parent=ss["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#27ae60"),
        leading=15,
    ))
    ss.add(ParagraphStyle(
        "ScoreMed",
        parent=ss["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#f39c12"),
        leading=15,
    ))
    ss.add(ParagraphStyle(
        "ScoreLow",
        parent=ss["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#e74c3c"),
        leading=15,
    ))
    return ss


def _score_style(score: int, ss):
    if score >= 70:
        return ss["ScoreHigh"]
    if score >= 40:
        return ss["ScoreMed"]
    return ss["ScoreLow"]


def _table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f7fa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f7fa"), colors.white]),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])


# ── Helpers ──────────────────────────────────────────────────────────────

def _format_duration(iso_dur: str) -> str:
    """Convert ISO 8601 duration like PT12H30M to '12h 30m'."""
    if not iso_dur:
        return "—"
    d = iso_dur.replace("PT", "").replace("H", "h ").replace("M", "m")
    return d.strip()


def _format_datetime(dt_str: str) -> str:
    """Format ISO datetime to readable Spanish format."""
    if not dt_str:
        return "—"
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%d/%m/%Y %H:%M")
    except (ValueError, TypeError):
        return dt_str


# ── Main builder ─────────────────────────────────────────────────────────


def generate_report(
    request: TravelRequest,
    destination_reports: list[DestinationReport],
    output_dir: str = "reports",
) -> str:
    """Build the comparative PDF and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"informe_viaje_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    ss = _styles()
    elements: list = []

    # ── Cover page ───────────────────────────────────────────────────
    elements.append(Spacer(1, 1.5 * inch))
    elements.append(Paragraph("Informe Comparativo de Viaje", ss["CoverTitle"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(HRFlowable(
        width="80%", thickness=2, color=colors.HexColor("#1a3c5e"),
        spaceAfter=20,
    ))
    elements.append(Paragraph(
        f"Generado el {datetime.now().strftime('%d de %B de %Y')}",
        ss["CoverSubtitle"],
    ))
    elements.append(Spacer(1, 0.3 * inch))

    # User preferences summary
    dates_str = ", ".join(
        f"{d.date_from.strftime('%d/%m/%Y')} – {d.date_to.strftime('%d/%m/%Y')}"
        for d in request.available_dates
    )
    summary_data = [
        ["Parámetro", "Valor"],
        ["Origen", request.origin],
        ["Clima preferido", request.preferred_climate],
        ["Región", request.region],
        ["Fechas disponibles", dates_str],
        ["Presupuesto máximo", f"{request.max_budget:,.0f} {request.currency} por persona"],
        ["Actividades", ", ".join(request.preferred_activities)],
        ["Viajeros", str(request.num_people)],
    ]
    t = Table(summary_data, colWidths=[2.2 * inch, 4.5 * inch])
    t.setStyle(_table_style())
    elements.append(t)
    elements.append(PageBreak())

    # ── Comparison table ─────────────────────────────────────────────
    elements.append(Paragraph("Tabla Comparativa de Destinos", ss["SectionHeader"]))
    elements.append(Spacer(1, 6 * mm))

    sorted_reports = sorted(destination_reports, key=lambda r: r.overall_score, reverse=True)

    comp_header = [
        "Destino", "País", "Precio\nmínimo", "Escalas\nmín.",
        "Clima", "Compatibilidad\nclima", "Compatibilidad\nactividades", "Puntuación\ngeneral",
    ]
    comp_data = [comp_header]
    for dr in sorted_reports:
        min_price = min((f.price for f in dr.flights), default=0)
        min_stops = min((f.stops for f in dr.flights), default=0)
        weather_desc = dr.weather.description if dr.weather else "N/A"
        comp_data.append([
            dr.destination.city,
            dr.destination.country,
            f"{min_price:,.0f} {request.currency}" if min_price else "N/A",
            str(min_stops) if dr.flights else "N/A",
            weather_desc,
            f"{dr.destination.climate_match}%",
            f"{dr.destination.activity_match}%",
            f"{dr.overall_score}%",
        ])

    comp_table = Table(comp_data, repeatRows=1)
    comp_table.setStyle(_table_style())
    elements.append(comp_table)
    elements.append(PageBreak())

    # ── Detail pages per destination ─────────────────────────────────
    for idx, dr in enumerate(sorted_reports, 1):
        elements.append(Paragraph(
            f"{idx}. {dr.destination.city}, {dr.destination.country}",
            ss["SectionHeader"],
        ))
        elements.append(Paragraph(
            f"<b>Razón de selección:</b> {dr.destination.reasoning}",
            ss["BodyText2"],
        ))
        elements.append(Paragraph(
            f"<b>Puntuación general:</b> {dr.overall_score}%",
            _score_style(dr.overall_score, ss),
        ))
        elements.append(Spacer(1, 4 * mm))

        # Weather
        if dr.weather and dr.weather.description != "Datos climáticos no disponibles":
            elements.append(Paragraph("Clima esperado", ss["SubHeader"]))
            weather_data = [
                ["Temp. media", "Temp. mín.", "Temp. máx.", "Precipitación", "Descripción"],
                [
                    f"{dr.weather.avg_temp_c}°C",
                    f"{dr.weather.min_temp_c}°C",
                    f"{dr.weather.max_temp_c}°C",
                    f"{dr.weather.avg_precipitation_mm} mm/día",
                    dr.weather.description,
                ],
            ]
            wt = Table(weather_data, colWidths=[1.1 * inch] * 4 + [2.3 * inch])
            wt.setStyle(_table_style())
            elements.append(wt)
            elements.append(Spacer(1, 4 * mm))

        # Flights
        if dr.flights:
            elements.append(Paragraph("Vuelos disponibles", ss["SubHeader"]))
            flight_header = ["Aerolínea", "Precio", "Salida", "Duración", "Escalas"]
            flight_data = [flight_header]
            for fl in dr.flights:
                flight_data.append([
                    fl.airline,
                    f"{fl.price:,.0f} {fl.currency}",
                    _format_datetime(fl.departure),
                    _format_duration(fl.duration),
                    str(fl.stops),
                ])
            ft = Table(flight_data, repeatRows=1)
            ft.setStyle(_table_style())
            elements.append(ft)
            elements.append(Spacer(1, 4 * mm))

        # Activities
        if dr.activities_description:
            elements.append(Paragraph("Actividades recomendadas", ss["SubHeader"]))
            for line in dr.activities_description.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Skip markdown headings (# lines) -- already have SubHeader above
                if re.match(r"^#{1,4}\s", line):
                    continue
                # Convert markdown bold **text** to ReportLab bold <b>text</b>
                line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
                # Strip leading bullet markers (-, *, bullet)
                line = re.sub(r"^[\-\*\u2022]\s*", "", line)
                # Indent bullet items with a dash prefix
                if line:
                    elements.append(Paragraph(f"- {line}", ss["BodyText2"]))
            elements.append(Spacer(1, 4 * mm))

        elements.append(HRFlowable(
            width="100%", thickness=1, color=colors.HexColor("#cccccc"),
            spaceAfter=10, spaceBefore=10,
        ))

        # Page break between destinations (except the last)
        if idx < len(sorted_reports):
            elements.append(PageBreak())

    # Build
    doc.build(elements)
    logger.info("PDF report generated: %s", filepath)
    return filepath
