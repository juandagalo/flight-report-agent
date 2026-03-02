"""Pydantic models for the API request / response."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class DateRange(BaseModel):
    """A travel date window."""

    date_from: date = Field(..., description="Fecha de salida (YYYY-MM-DD)")
    date_to: date = Field(..., description="Fecha de regreso (YYYY-MM-DD)")


class TravelRequest(BaseModel):
    """User-supplied travel preferences."""

    preferred_climate: str = Field(
        ...,
        description="Clima preferido: tropical, templado, frío, seco, etc.",
        examples=["tropical", "templado"],
    )
    region: str = Field(
        ...,
        description="Región del mundo de interés",
        examples=["Caribe", "Europa", "Sudamérica"],
    )
    available_dates: list[DateRange] = Field(
        ...,
        min_length=1,
        description="Ventanas de fechas disponibles para viajar",
    )
    max_budget: float = Field(
        ..., gt=0, description="Presupuesto máximo por persona en la moneda indicada"
    )
    currency: str = Field(
        default="USD", description="Moneda del presupuesto (ISO 4217)"
    )
    origin: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Código IATA del aeropuerto de origen",
        examples=["BOG", "MAD", "MIA"],
    )
    preferred_activities: list[str] = Field(
        ...,
        min_length=1,
        description="Actividades preferidas",
        examples=[["playa", "cultura", "gastronomía"]],
    )
    num_people: int = Field(..., ge=1, le=9, description="Número de viajeros")


class IntakeResult(TravelRequest):
    """Structured output from the intake agent — extends TravelRequest with assumptions."""

    assumptions: list[str] = Field(
        default_factory=list,
        description="Suposiciones hechas al interpretar el mensaje del usuario",
    )


# ----- Intermediate / internal models -----


class CandidateDestination(BaseModel):
    """A single destination suggested by the LLM."""

    city: str
    iata_code: str
    country: str
    reasoning: str = ""
    climate_match: int = Field(
        default=0, ge=0, le=100, description="Porcentaje de compatibilidad climática"
    )
    activity_match: int = Field(
        default=0, ge=0, le=100, description="Porcentaje de compatibilidad de actividades"
    )


class FlightOffer(BaseModel):
    """A single flight offer from Amadeus."""

    airline: str = ""
    price: float = 0.0
    currency: str = "USD"
    departure: str = ""
    arrival: str = ""
    duration: str = ""
    stops: int = 0
    return_departure: Optional[str] = None
    return_arrival: Optional[str] = None
    return_duration: Optional[str] = None
    return_stops: Optional[int] = None


class WeatherInfo(BaseModel):
    """Average weather for a destination during the travel window."""

    avg_temp_c: float = 0.0
    min_temp_c: float = 0.0
    max_temp_c: float = 0.0
    avg_precipitation_mm: float = 0.0
    description: str = ""


class DestinationReport(BaseModel):
    """Full enriched data for one destination."""

    destination: CandidateDestination
    flights: list[FlightOffer] = []
    weather: Optional[WeatherInfo] = None
    activities_description: str = ""
    overall_score: int = Field(
        default=0, ge=0, le=100, description="Puntuación general de compatibilidad"
    )
