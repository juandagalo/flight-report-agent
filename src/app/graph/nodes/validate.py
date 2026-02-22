"""Node: Validate user input."""

from __future__ import annotations

import logging

from src.app.graph.state import TravelState

logger = logging.getLogger(__name__)

# Common IATA codes for quick validation
KNOWN_IATA = {
    "BOG", "MDE", "CLO", "CTG", "MAD", "BCN", "LIS", "CDG", "FCO", "LHR",
    "AMS", "FRA", "MUC", "MIA", "JFK", "LAX", "ORD", "ATL", "DFW", "EWR",
    "CUN", "MEX", "GDL", "PTY", "SJO", "LIM", "SCL", "EZE", "GRU", "GIG",
    "UIO", "HAV", "SDQ", "PUJ", "SJU", "MBJ", "NAS", "BKK", "NRT", "HND",
    "ICN", "SIN", "DPS", "MLE", "DXB", "DOH", "IST", "ATH", "VIE", "ZRH",
    "CPH", "OSL", "ARN", "HEL", "WAW", "PRG", "BUD",
}


def validate_input(state: TravelState) -> TravelState:
    """Validate and normalise the request data."""

    logger.info("→ Starting VALIDATE node")

    errors: list[str] = []
    request = state.get("request")
    if request is None:
        return {**state, "validated": False, "validation_errors": ["No request found in state"], "errors": ["No request found in state"]}

    # Origin IATA
    origin = request.origin.upper().strip()
    if len(origin) != 3:
        errors.append(f"El código IATA de origen '{origin}' debe tener 3 caracteres.")
        
    if origin not in KNOWN_IATA:
        logger.warning("IATA code '%s' not in known list – accepting anyway", origin)

    # Dates
    for dr in request.available_dates:
        if dr.date_to <= dr.date_from:
            errors.append(
                f"La fecha de regreso ({dr.date_to}) debe ser posterior a la de salida ({dr.date_from})."
            )

    # Budget
    if request.max_budget <= 0:
        errors.append("El presupuesto debe ser mayor a 0.")

    # People
    if request.num_people < 1 or request.num_people > 9:
        errors.append("El número de viajeros debe estar entre 1 y 9.")

    if errors:
        logger.warning("Validation errors: %s", errors)

    return {
        **state,
        "validated": len(errors) == 0,
        "validation_errors": errors,
        "errors": state.get("errors", []) + errors,
        "suggest_retry_count": state.get("suggest_retry_count", 0),
    }
