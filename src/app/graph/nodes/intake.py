"""Node: Convert natural-language user message into a structured TravelRequest."""

from __future__ import annotations

import logging

from src.app.graph.state import TravelState
from src.app.prompts.templates import INTAKE_SYSTEM, INTAKE_USER
from src.app.schemas import IntakeResult, TravelRequest
from src.app.services.llm_provider import get_llm

logger = logging.getLogger(__name__)


async def intake_node(state: TravelState) -> dict:
    """Extract travel preferences from a natural-language message.

    If ``user_message`` is empty (i.e. the caller already supplied a structured
    ``request``), this node is a no-op passthrough for backwards compatibility.
    """

    user_message = state.get("user_message", "")

    if not user_message:
        logger.info("→ INTAKE node: no user_message, passthrough")
        return {}

    logger.info("→ Starting INTAKE node")

    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(IntakeResult, method="json_mode")

    user_msg = INTAKE_USER.format(message=user_message)

    try:
        result: IntakeResult = await structured_llm.ainvoke([
            {"role": "system", "content": INTAKE_SYSTEM},
            {"role": "user", "content": user_msg},
        ])

        # Build a plain TravelRequest (without the assumptions field)
        travel_request = TravelRequest(**{
            k: v for k, v in result.model_dump().items()
            if k != "assumptions"
        })

        logger.info(
            "Intake extracted: origin=%s, region=%s, budget=%s %s, assumptions=%s",
            travel_request.origin,
            travel_request.region,
            travel_request.max_budget,
            travel_request.currency,
            result.assumptions,
        )

        return {
            "request": travel_request,
            "intake_assumptions": result.assumptions,
        }

    except Exception as exc:
        logger.error("Intake node failed: %s", exc)
        return {
            "errors": [f"Error al interpretar el mensaje: {exc}"],
        }
