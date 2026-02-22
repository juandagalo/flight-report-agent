"""Open-Meteo weather client — free, no API key required."""

from __future__ import annotations

import logging
from datetime import date

import httpx

from src.app.schemas import WeatherInfo

logger = logging.getLogger(__name__)

# Simple mapping of popular cities to (lat, lon).
# The LLM node can also supply coordinates; this is a fallback.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "CUN": (21.17, -86.85),   # Cancún
    "PTY": (8.97, -79.53),    # Panamá City
    "SJO": (9.99, -84.21),    # San José, CR
    "HAV": (23.14, -82.36),   # La Habana
    "SDQ": (18.47, -69.88),   # Santo Domingo
    "SJU": (18.44, -66.00),   # San Juan, PR
    "LIS": (38.72, -9.14),    # Lisboa
    "BCN": (41.39, 2.16),     # Barcelona
    "MAD": (40.42, -3.70),    # Madrid
    "FCO": (41.90, 12.50),    # Roma
    "CDG": (48.86, 2.35),     # París
    "LHR": (51.51, -0.13),    # Londres
    "MIA": (25.76, -80.19),   # Miami
    "BOG": (4.71, -74.07),    # Bogotá
    "LIM": (12.05, -77.04),   # Lima
    "SCL": (-33.45, -70.67),  # Santiago
    "EZE": (-34.60, -58.38),  # Buenos Aires
    "GRU": (-23.55, -46.63),  # São Paulo
    "NRT": (35.68, 139.69),   # Tokio
    "BKK": (13.76, 100.50),   # Bangkok
    "DPS": (-8.65, 115.22),   # Bali
    "MLE": (4.18, 73.51),     # Maldivas
    "JFK": (40.71, -74.01),   # New York
    "LAX": (34.05, -118.24),  # Los Angeles
    "ATH": (37.97, 23.73),    # Atenas
    "IST": (41.01, 28.98),    # Estambul
    "MBJ": (18.50, -77.92),   # Montego Bay
    "PUJ": (18.57, -68.37),   # Punta Cana
    "CTG": (10.39, -75.51),   # Cartagena
    "MEX": (19.43, -99.13),   # Ciudad de México
    "UIO": (0.18, -78.47),    # Quito
}


async def get_weather(
    iata_code: str,
    date_from: date,
    date_to: date,
    lat: float | None = None,
    lon: float | None = None,
) -> WeatherInfo:
    """Fetch historical climate averages from Open-Meteo for the given period.

    Uses an *archive* endpoint, which provides past-year averages for the
    same calendar window — a reasonable proxy for expected weather.
    """
    if lat is None or lon is None:
        coords = CITY_COORDS.get(iata_code)
        if coords is None:
            logger.warning("No coordinates for %s – returning empty weather", iata_code)
            return WeatherInfo(description="Datos climáticos no disponibles")
        lat, lon = coords

    # Use previous year's actual data as a climate proxy
    hist_from = date_from.replace(year=date_from.year - 1)
    hist_to = date_to.replace(year=date_to.year - 1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": hist_from.isoformat(),
        "end_date": hist_to.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        daily = data.get("daily", {})
        temps_mean = daily.get("temperature_2m_mean", [])
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])

        avg_temp = sum(temps_mean) / len(temps_mean) if temps_mean else 0
        avg_max = sum(temps_max) / len(temps_max) if temps_max else 0
        avg_min = sum(temps_min) / len(temps_min) if temps_min else 0
        avg_precip = sum(precip) / len(precip) if precip else 0

        # Build a human-readable Spanish description
        if avg_temp > 25:
            desc = "Cálido y tropical"
        elif avg_temp > 18:
            desc = "Templado y agradable"
        elif avg_temp > 10:
            desc = "Fresco"
        else:
            desc = "Frío"

        if avg_precip > 5:
            desc += ", lluvioso"
        elif avg_precip > 1:
            desc += ", lluvias ocasionales"
        else:
            desc += ", seco"

        return WeatherInfo(
            avg_temp_c=round(avg_temp, 1),
            min_temp_c=round(avg_min, 1),
            max_temp_c=round(avg_max, 1),
            avg_precipitation_mm=round(avg_precip, 1),
            description=desc,
        )
    except Exception as exc:
        logger.error("Open-Meteo error for %s: %s", iata_code, exc)
        return WeatherInfo(description="Datos climáticos no disponibles")
