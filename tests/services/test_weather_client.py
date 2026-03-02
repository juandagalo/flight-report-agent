"""Tests for the weather client — mocks httpx.AsyncClient."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from src.app.services.weather_client import get_weather, CITY_COORDS


def _mock_response(avg_temp: float = 28.0, precip: float = 3.0) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "daily": {
            "temperature_2m_mean": [avg_temp] * 5,
            "temperature_2m_max": [avg_temp + 3] * 5,
            "temperature_2m_min": [avg_temp - 3] * 5,
            "precipitation_sum": [precip] * 5,
        }
    }
    return resp


class TestGetWeather:
    @patch("src.app.services.weather_client.httpx.AsyncClient")
    async def test_known_city(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(avg_temp=28.0, precip=3.0))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await get_weather("CUN", date(2026, 7, 1), date(2026, 7, 15))
        assert result.avg_temp_c == 28.0
        assert "Cálido" in result.description

    async def test_unknown_city_fallback(self):
        result = await get_weather("ZZZ", date(2026, 7, 1), date(2026, 7, 15))
        assert result.description == "Datos climáticos no disponibles"

    @patch("src.app.services.weather_client.httpx.AsyncClient")
    async def test_explicit_coords(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(avg_temp=15.0, precip=0.5))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await get_weather("ZZZ", date(2026, 7, 1), date(2026, 7, 15), lat=40.0, lon=-3.0)
        assert result.avg_temp_c == 15.0
        assert "Fresco" in result.description

    @patch("src.app.services.weather_client.httpx.AsyncClient")
    async def test_hot_rainy_description(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response(avg_temp=30.0, precip=8.0))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await get_weather("CUN", date(2026, 7, 1), date(2026, 7, 15))
        assert "lluvioso" in result.description

    @patch("src.app.services.weather_client.httpx.AsyncClient")
    async def test_http_error_fallback(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await get_weather("CUN", date(2026, 7, 1), date(2026, 7, 15))
        assert result.description == "Datos climáticos no disponibles"

    @patch("src.app.services.weather_client.httpx.AsyncClient")
    async def test_previous_year_dates(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_response())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        await get_weather("BOG", date(2026, 7, 1), date(2026, 7, 15))

        call_kwargs = mock_client.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        assert params["start_date"] == "2025-07-01"
        assert params["end_date"] == "2025-07-15"
