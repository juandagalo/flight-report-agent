"""End-to-end integration tests for the full pipeline with RAG context.

Uses a real Qdrant instance in embedded mode (tmp_path) but mocks all
external APIs: LLM (get_llm), embeddings (embed_texts, embed_query),
Amadeus flight search, and Open-Meteo weather.
"""

from __future__ import annotations

import random
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.schemas import (
    CandidateDestination,
    DateRange,
    DestinationReport,
    FlightOffer,
    IntakeResult,
    TravelRequest,
    WeatherInfo,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1536


def _random_vector(seed: int = 0) -> list[float]:
    """Return a deterministic pseudo-random unit vector for testing."""
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


# Pre-computed vectors so searches actually return results based on
# cosine similarity.
VEC_BEACH = _random_vector(seed=42)
VEC_CULTURE = _random_vector(seed=43)
VEC_QUERY = _random_vector(seed=42)  # Same as VEC_BEACH to ensure match


# ---------------------------------------------------------------------------
# Test chunks to seed into Qdrant
# ---------------------------------------------------------------------------

SEED_CHUNKS = [
    {
        "id": "CUN_chunk_001",
        "vector": VEC_BEACH,
        "payload": {
            "text": "Cancun has beautiful white-sand beaches and warm Caribbean waters year round.",
            "heading": "See",
            "city": "Cancun",
            "country": "Mexico",
            "iata": "CUN",
        },
    },
    {
        "id": "CUN_chunk_002",
        "vector": _random_vector(seed=44),
        "payload": {
            "text": "The Hotel Zone in Cancun stretches along a narrow strip with resorts and nightlife.",
            "heading": "Sleep",
            "city": "Cancun",
            "country": "Mexico",
            "iata": "CUN",
        },
    },
    {
        "id": "BCN_chunk_001",
        "vector": VEC_CULTURE,
        "payload": {
            "text": "Barcelona is famous for Gaudi architecture, La Rambla, and vibrant cultural life.",
            "heading": "See",
            "city": "Barcelona",
            "country": "Spain",
            "iata": "BCN",
        },
    },
    {
        "id": "BCN_chunk_002",
        "vector": _random_vector(seed=45),
        "payload": {
            "text": "Barcelona offers tapas tours, flamenco shows, and Gothic Quarter walks.",
            "heading": "Do",
            "city": "Barcelona",
            "country": "Spain",
            "iata": "BCN",
        },
    },
    {
        "id": "PUJ_chunk_001",
        "vector": _random_vector(seed=46),
        "payload": {
            "text": "Punta Cana is known for all-inclusive resorts and clear turquoise waters.",
            "heading": "See",
            "city": "Punta Cana",
            "country": "Dominican Republic",
            "iata": "PUJ",
        },
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def qdrant_tmp(tmp_path, monkeypatch):
    """Point Qdrant to a temp directory and reset client between tests."""
    monkeypatch.setattr(
        "src.app.services.qdrant_client.settings",
        MagicMock(
            QDRANT_PATH=str(tmp_path / "qdrant"),
            QDRANT_COLLECTION_KNOWLEDGE="travel_knowledge",
            QDRANT_COLLECTION_INTERACTIONS="interactions",
            EMBEDDING_DIMENSION=EMBEDDING_DIM,
        ),
    )
    # Also patch the settings reference in rag.py and store_interaction.py
    monkeypatch.setattr(
        "src.app.services.rag.settings",
        MagicMock(
            QDRANT_COLLECTION_KNOWLEDGE="travel_knowledge",
            QDRANT_COLLECTION_INTERACTIONS="interactions",
        ),
    )
    monkeypatch.setattr(
        "src.app.graph.nodes.store_interaction.settings",
        MagicMock(
            QDRANT_COLLECTION_INTERACTIONS="interactions",
        ),
    )
    from src.app.services.qdrant_client import reset_client

    reset_client()
    yield
    reset_client()


@pytest.fixture()
def seeded_qdrant(qdrant_tmp):
    """Create collections and seed travel_knowledge with test chunks."""
    from src.app.services.qdrant_client import (
        ensure_all_collections,
        upsert_points,
    )

    ensure_all_collections()
    upsert_points(
        collection_name="travel_knowledge",
        ids=[c["id"] for c in SEED_CHUNKS],
        vectors=[c["vector"] for c in SEED_CHUNKS],
        payloads=[c["payload"] for c in SEED_CHUNKS],
    )
    return None


@pytest.fixture()
def sample_request() -> TravelRequest:
    return TravelRequest(
        preferred_climate="tropical",
        region="Caribe",
        available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
        max_budget=1500.0,
        currency="USD",
        origin="BOG",
        preferred_activities=["playa", "cultura"],
        num_people=2,
    )


@pytest.fixture()
def sample_intake_result() -> IntakeResult:
    return IntakeResult(
        preferred_climate="tropical",
        region="Caribe",
        available_dates=[DateRange(date_from=date(2026, 7, 1), date_to=date(2026, 7, 15))],
        max_budget=1500.0,
        currency="USD",
        origin="BOG",
        preferred_activities=["playa", "cultura"],
        num_people=2,
        assumptions=["Assumed July 1-15 as travel dates"],
    )


def _mock_destinations():
    """Destination list that the mock LLM returns from suggest node."""
    from src.app.graph.nodes.suggest import DestinationList

    return DestinationList(
        destinations=[
            CandidateDestination(
                city="Cancun",
                iata_code="CUN",
                country="Mexico",
                reasoning="Caribbean beach destination",
                climate_match=90,
                activity_match=80,
            ),
            CandidateDestination(
                city="Punta Cana",
                iata_code="PUJ",
                country="Dominican Republic",
                reasoning="All-inclusive beach resort",
                climate_match=88,
                activity_match=75,
            ),
        ]
    )


def _mock_flight_offer(dest_code: str, price: float = 800.0) -> FlightOffer:
    return FlightOffer(
        airline="TestAir",
        price=price,
        currency="USD",
        departure="2026-07-01T08:00:00",
        arrival="2026-07-01T12:00:00",
        duration="PT4H0M",
        stops=0,
        return_departure="2026-07-15T14:00:00",
        return_arrival="2026-07-15T18:00:00",
        return_duration="PT4H0M",
        return_stops=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipelineWithRag:
    """Full pipeline with mocked externals and real Qdrant (seeded)."""

    @patch("src.app.graph.nodes.store_interaction.embed_texts")
    @patch("src.app.graph.nodes.enrich.query_travel_knowledge")
    @patch("src.app.graph.nodes.suggest.query_interactions")
    @patch("src.app.graph.nodes.suggest.query_travel_knowledge")
    @patch("src.app.services.rag.embed_query")
    @patch("src.app.graph.nodes.enrich.get_weather")
    @patch("src.app.graph.nodes.search_flights.search_flights")
    @patch("src.app.graph.nodes.report.generate_report")
    @patch("src.app.graph.nodes.intake.get_llm")
    @patch("src.app.graph.nodes.suggest.get_llm")
    @patch("src.app.graph.nodes.enrich.get_llm")
    async def test_full_pipeline_with_rag_context(
        self,
        mock_enrich_llm,
        mock_suggest_llm,
        mock_intake_llm,
        mock_generate_report,
        mock_search_flights,
        mock_get_weather,
        mock_rag_embed_query,
        mock_suggest_query_knowledge,
        mock_suggest_query_interactions,
        mock_enrich_query_knowledge,
        mock_store_embed_texts,
        seeded_qdrant,
        sample_intake_result,
        tmp_path,
    ):
        """Run the full pipeline with seeded Qdrant -- verify RAG context
        is fetched and used in suggest and enrich nodes."""

        # -- Intake LLM mock --
        intake_structured = AsyncMock(ainvoke=AsyncMock(return_value=sample_intake_result))
        intake_mock = MagicMock()
        intake_mock.with_structured_output.return_value = intake_structured
        mock_intake_llm.return_value = intake_mock

        # -- Suggest LLM mock --
        suggest_structured = AsyncMock(ainvoke=AsyncMock(return_value=_mock_destinations()))
        suggest_mock = MagicMock()
        suggest_mock.with_structured_output.return_value = suggest_structured
        mock_suggest_llm.return_value = suggest_mock

        # -- Suggest RAG mocks: return real-looking results --
        mock_suggest_query_knowledge.return_value = [
            {
                "id": "CUN_chunk_001",
                "score": 0.85,
                "payload": {
                    "text": "Cancun has beautiful beaches",
                    "heading": "See",
                    "city": "Cancun",
                    "iata": "CUN",
                },
            }
        ]
        mock_suggest_query_interactions.return_value = []

        # -- Enrich RAG mock --
        mock_enrich_query_knowledge.return_value = [
            {
                "id": "CUN_chunk_002",
                "score": 0.80,
                "payload": {
                    "text": "Hotel Zone nightlife",
                    "heading": "Sleep",
                    "city": "Cancun",
                    "iata": "CUN",
                },
            }
        ]

        # -- Enrich LLM mock --
        enrich_resp = MagicMock(content="-- Actividad 1: Snorkel\n-- Actividad 2: Tour")
        enrich_llm = MagicMock()
        enrich_llm.ainvoke = AsyncMock(return_value=enrich_resp)
        mock_enrich_llm.return_value = enrich_llm

        # -- Amadeus mock: return affordable flights for both destinations --
        def _search_flights_side_effect(
            origin, destination, departure_date, return_date,
            adults=1, currency="USD", max_results=3,
        ):
            return [_mock_flight_offer(destination, price=800.0)]

        mock_search_flights.side_effect = _search_flights_side_effect

        # -- Weather mock --
        mock_get_weather.return_value = WeatherInfo(
            avg_temp_c=28.5, min_temp_c=24.0, max_temp_c=33.0,
            avg_precipitation_mm=3.2, description="Warm and tropical",
        )

        # -- Report mock --
        report_path = str(tmp_path / "test_report.pdf")
        mock_generate_report.return_value = report_path

        # -- Store interaction mock --
        mock_store_embed_texts.return_value = [_random_vector(seed=99)]

        # -- Build and run the pipeline --
        from src.app.graph.pipeline import build_graph

        graph = build_graph()
        initial_state = {
            "user_message": "Quiero ir a la playa desde Bogota en julio, presupuesto 1500 USD para 2 personas",
            "user_id": "",
            "validated": False,
            "validation_errors": [],
            "candidate_destinations": [],
            "destination_reports": [],
            "enriched": False,
            "report_path": "",
            "suggest_retry_count": 0,
            "errors": [],
        }

        result = await graph.ainvoke(initial_state)

        # -- Assertions --
        # Pipeline completed and produced a report
        assert result.get("report_path") == report_path
        assert result.get("enriched") is True
        assert result.get("validated") is True

        # Suggest node used RAG context (knowledge was called)
        mock_suggest_query_knowledge.assert_called_once()

        # Enrich node used RAG context for each destination
        assert mock_enrich_query_knowledge.call_count == 2  # CUN and PUJ

        # Destination reports are enriched
        reports = result.get("destination_reports", [])
        assert len(reports) == 2
        for r in reports:
            assert r.weather is not None
            assert r.activities_description != ""
            assert r.overall_score > 0

        # Suggest LLM was called with RAG-enhanced prompt (system msg includes
        # the RAG keywords)
        suggest_call_args = suggest_structured.ainvoke.call_args[0][0]
        system_content = suggest_call_args[0]["content"]
        assert "informacion de contexto" in system_content.lower() or "contexto" in system_content.lower()

    @patch("src.app.graph.nodes.store_interaction.embed_texts")
    @patch("src.app.graph.nodes.enrich.query_travel_knowledge")
    @patch("src.app.graph.nodes.suggest.query_interactions")
    @patch("src.app.graph.nodes.suggest.query_travel_knowledge")
    @patch("src.app.graph.nodes.enrich.get_weather")
    @patch("src.app.graph.nodes.search_flights.search_flights")
    @patch("src.app.graph.nodes.report.generate_report")
    @patch("src.app.graph.nodes.intake.get_llm")
    @patch("src.app.graph.nodes.suggest.get_llm")
    @patch("src.app.graph.nodes.enrich.get_llm")
    async def test_full_pipeline_without_rag_fallback(
        self,
        mock_enrich_llm,
        mock_suggest_llm,
        mock_intake_llm,
        mock_generate_report,
        mock_search_flights,
        mock_get_weather,
        mock_suggest_query_knowledge,
        mock_suggest_query_interactions,
        mock_enrich_query_knowledge,
        mock_store_embed_texts,
        qdrant_tmp,
        sample_intake_result,
        tmp_path,
    ):
        """Pipeline with empty Qdrant -- RAG returns empty, falls back to
        original templates. Pipeline still completes successfully."""

        # -- Intake LLM mock --
        intake_structured = AsyncMock(ainvoke=AsyncMock(return_value=sample_intake_result))
        intake_mock = MagicMock()
        intake_mock.with_structured_output.return_value = intake_structured
        mock_intake_llm.return_value = intake_mock

        # -- Suggest LLM mock --
        suggest_structured = AsyncMock(ainvoke=AsyncMock(return_value=_mock_destinations()))
        suggest_mock = MagicMock()
        suggest_mock.with_structured_output.return_value = suggest_structured
        mock_suggest_llm.return_value = suggest_mock

        # -- RAG returns empty (no collections seeded) --
        mock_suggest_query_knowledge.return_value = []
        mock_suggest_query_interactions.return_value = []
        mock_enrich_query_knowledge.return_value = []

        # -- Enrich LLM mock --
        enrich_resp = MagicMock(content="-- Actividad sin RAG")
        enrich_llm = MagicMock()
        enrich_llm.ainvoke = AsyncMock(return_value=enrich_resp)
        mock_enrich_llm.return_value = enrich_llm

        # -- Amadeus mock --
        mock_search_flights.side_effect = lambda *a, **kw: [_mock_flight_offer("TST")]

        # -- Weather mock --
        mock_get_weather.return_value = WeatherInfo(
            avg_temp_c=25.0, description="Templado",
        )

        # -- Report mock --
        report_path = str(tmp_path / "test_report_no_rag.pdf")
        mock_generate_report.return_value = report_path

        # -- Store interaction: no user_id so it will skip --
        mock_store_embed_texts.return_value = [_random_vector(seed=99)]

        from src.app.graph.pipeline import build_graph

        graph = build_graph()
        initial_state = {
            "user_message": "Quiero ir a la playa desde Bogota en julio",
            "user_id": "",
            "validated": False,
            "validation_errors": [],
            "candidate_destinations": [],
            "destination_reports": [],
            "enriched": False,
            "report_path": "",
            "suggest_retry_count": 0,
            "errors": [],
        }

        result = await graph.ainvoke(initial_state)

        # Pipeline completed successfully
        assert result.get("report_path") == report_path
        assert result.get("enriched") is True
        assert result.get("validated") is True

        # Suggest used the non-RAG (original) templates since RAG returned empty
        suggest_call_args = suggest_structured.ainvoke.call_args[0][0]
        system_content = suggest_call_args[0]["content"]
        # Original template does NOT contain "informacion de contexto proporcionada"
        assert "informacion de contexto proporcionada" not in system_content.lower()

        # Enrich used non-RAG template (no rag_context placeholder filled)
        enrich_call_args = enrich_llm.ainvoke.call_args[0][0]
        user_content = enrich_call_args[1]["content"]
        assert "--- Destination Info ---" not in user_content


class TestInteractionStorageRoundTrip:
    """Verify that store_interaction actually persists data to Qdrant."""

    @patch("src.app.graph.nodes.store_interaction.embed_texts")
    @patch("src.app.graph.nodes.enrich.query_travel_knowledge")
    @patch("src.app.graph.nodes.suggest.query_interactions")
    @patch("src.app.graph.nodes.suggest.query_travel_knowledge")
    @patch("src.app.graph.nodes.enrich.get_weather")
    @patch("src.app.graph.nodes.search_flights.search_flights")
    @patch("src.app.graph.nodes.report.generate_report")
    @patch("src.app.graph.nodes.intake.get_llm")
    @patch("src.app.graph.nodes.suggest.get_llm")
    @patch("src.app.graph.nodes.enrich.get_llm")
    async def test_interaction_stored_after_pipeline(
        self,
        mock_enrich_llm,
        mock_suggest_llm,
        mock_intake_llm,
        mock_generate_report,
        mock_search_flights,
        mock_get_weather,
        mock_suggest_query_knowledge,
        mock_suggest_query_interactions,
        mock_enrich_query_knowledge,
        mock_store_embed_texts,
        qdrant_tmp,
        sample_intake_result,
        tmp_path,
    ):
        """Run pipeline with user_id='test-user' and verify that the
        interactions collection has a new point stored."""

        # -- Intake LLM mock --
        intake_structured = AsyncMock(ainvoke=AsyncMock(return_value=sample_intake_result))
        intake_mock = MagicMock()
        intake_mock.with_structured_output.return_value = intake_structured
        mock_intake_llm.return_value = intake_mock

        # -- Suggest LLM mock --
        suggest_structured = AsyncMock(ainvoke=AsyncMock(return_value=_mock_destinations()))
        suggest_mock = MagicMock()
        suggest_mock.with_structured_output.return_value = suggest_structured
        mock_suggest_llm.return_value = suggest_mock

        # -- RAG returns empty --
        mock_suggest_query_knowledge.return_value = []
        mock_suggest_query_interactions.return_value = []
        mock_enrich_query_knowledge.return_value = []

        # -- Enrich LLM mock --
        enrich_resp = MagicMock(content="-- Actividades de prueba")
        enrich_llm = MagicMock()
        enrich_llm.ainvoke = AsyncMock(return_value=enrich_resp)
        mock_enrich_llm.return_value = enrich_llm

        # -- Amadeus mock --
        mock_search_flights.side_effect = lambda *a, **kw: [_mock_flight_offer("TST")]

        # -- Weather mock --
        mock_get_weather.return_value = WeatherInfo(avg_temp_c=28.0, description="Warm")

        # -- Report mock --
        report_path = str(tmp_path / "test_interaction_report.pdf")
        mock_generate_report.return_value = report_path

        # -- Store interaction embedding mock --
        interaction_vector = _random_vector(seed=100)
        mock_store_embed_texts.return_value = [interaction_vector]

        from src.app.graph.pipeline import build_graph

        graph = build_graph()
        initial_state = {
            "user_message": "Beach trip from Bogota",
            "user_id": "test-user",
            "validated": False,
            "validation_errors": [],
            "candidate_destinations": [],
            "destination_reports": [],
            "enriched": False,
            "report_path": "",
            "suggest_retry_count": 0,
            "errors": [],
        }

        result = await graph.ainvoke(initial_state)

        # Pipeline completed
        assert result.get("report_path") == report_path

        # Verify embed_texts was called (interaction was embedded)
        mock_store_embed_texts.assert_called_once()
        call_args = mock_store_embed_texts.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert "test-user" not in call_args[0]  # The summary text, not the user_id
        assert "BOG" in call_args[0] or "Caribe" in call_args[0]

        # Verify upsert was called -- the interaction should be in Qdrant
        from src.app.services.qdrant_client import get_client

        client = get_client()
        collection_info = client.get_collection("interactions")
        assert collection_info.points_count == 1

    @patch("src.app.graph.nodes.store_interaction.embed_texts")
    @patch("src.app.graph.nodes.enrich.query_travel_knowledge")
    @patch("src.app.graph.nodes.suggest.query_interactions")
    @patch("src.app.graph.nodes.suggest.query_travel_knowledge")
    @patch("src.app.graph.nodes.enrich.get_weather")
    @patch("src.app.graph.nodes.search_flights.search_flights")
    @patch("src.app.graph.nodes.report.generate_report")
    @patch("src.app.graph.nodes.intake.get_llm")
    @patch("src.app.graph.nodes.suggest.get_llm")
    @patch("src.app.graph.nodes.enrich.get_llm")
    async def test_no_interaction_stored_without_user_id(
        self,
        mock_enrich_llm,
        mock_suggest_llm,
        mock_intake_llm,
        mock_generate_report,
        mock_search_flights,
        mock_get_weather,
        mock_suggest_query_knowledge,
        mock_suggest_query_interactions,
        mock_enrich_query_knowledge,
        mock_store_embed_texts,
        qdrant_tmp,
        sample_intake_result,
        tmp_path,
    ):
        """Without user_id, store_interaction should skip and not persist
        anything to the interactions collection."""

        # -- Intake LLM mock --
        intake_structured = AsyncMock(ainvoke=AsyncMock(return_value=sample_intake_result))
        intake_mock = MagicMock()
        intake_mock.with_structured_output.return_value = intake_structured
        mock_intake_llm.return_value = intake_mock

        # -- Suggest LLM mock --
        suggest_structured = AsyncMock(ainvoke=AsyncMock(return_value=_mock_destinations()))
        suggest_mock = MagicMock()
        suggest_mock.with_structured_output.return_value = suggest_structured
        mock_suggest_llm.return_value = suggest_mock

        # -- RAG returns empty --
        mock_suggest_query_knowledge.return_value = []
        mock_suggest_query_interactions.return_value = []
        mock_enrich_query_knowledge.return_value = []

        # -- Enrich LLM mock --
        enrich_resp = MagicMock(content="-- Activities")
        enrich_llm = MagicMock()
        enrich_llm.ainvoke = AsyncMock(return_value=enrich_resp)
        mock_enrich_llm.return_value = enrich_llm

        # -- Amadeus mock --
        mock_search_flights.side_effect = lambda *a, **kw: [_mock_flight_offer("TST")]

        # -- Weather mock --
        mock_get_weather.return_value = WeatherInfo(avg_temp_c=25.0, description="Warm")

        # -- Report mock --
        report_path = str(tmp_path / "no_user_report.pdf")
        mock_generate_report.return_value = report_path

        from src.app.graph.pipeline import build_graph

        graph = build_graph()
        initial_state = {
            "user_message": "Trip from Bogota",
            "user_id": "",  # No user_id
            "validated": False,
            "validation_errors": [],
            "candidate_destinations": [],
            "destination_reports": [],
            "enriched": False,
            "report_path": "",
            "suggest_retry_count": 0,
            "errors": [],
        }

        result = await graph.ainvoke(initial_state)

        # Pipeline completed
        assert result.get("report_path") == report_path

        # embed_texts should NOT have been called (skipped due to no user_id)
        mock_store_embed_texts.assert_not_called()


class TestPipelineWithClaudeProvider:
    """Verify that the pipeline uses the get_llm factory, not ChatOpenAI directly."""

    @patch("src.app.graph.nodes.store_interaction.embed_texts")
    @patch("src.app.graph.nodes.enrich.query_travel_knowledge")
    @patch("src.app.graph.nodes.suggest.query_interactions")
    @patch("src.app.graph.nodes.suggest.query_travel_knowledge")
    @patch("src.app.graph.nodes.enrich.get_weather")
    @patch("src.app.graph.nodes.search_flights.search_flights")
    @patch("src.app.graph.nodes.report.generate_report")
    @patch("src.app.graph.nodes.intake.get_llm")
    @patch("src.app.graph.nodes.suggest.get_llm")
    @patch("src.app.graph.nodes.enrich.get_llm")
    async def test_pipeline_uses_get_llm_factory(
        self,
        mock_enrich_llm,
        mock_suggest_llm,
        mock_intake_llm,
        mock_generate_report,
        mock_search_flights,
        mock_get_weather,
        mock_suggest_query_knowledge,
        mock_suggest_query_interactions,
        mock_enrich_query_knowledge,
        mock_store_embed_texts,
        qdrant_tmp,
        sample_intake_result,
        tmp_path,
    ):
        """Mock get_llm at each node import location. Verify all three
        node modules call get_llm() -- proving the factory abstraction works
        regardless of which LLM_PROVIDER is configured."""

        # -- Intake LLM mock --
        intake_structured = AsyncMock(ainvoke=AsyncMock(return_value=sample_intake_result))
        intake_mock = MagicMock()
        intake_mock.with_structured_output.return_value = intake_structured
        mock_intake_llm.return_value = intake_mock

        # -- Suggest LLM mock --
        suggest_structured = AsyncMock(ainvoke=AsyncMock(return_value=_mock_destinations()))
        suggest_mock = MagicMock()
        suggest_mock.with_structured_output.return_value = suggest_structured
        mock_suggest_llm.return_value = suggest_mock

        # -- RAG returns empty --
        mock_suggest_query_knowledge.return_value = []
        mock_suggest_query_interactions.return_value = []
        mock_enrich_query_knowledge.return_value = []

        # -- Enrich LLM mock --
        enrich_resp = MagicMock(content="-- Test activities")
        enrich_llm = MagicMock()
        enrich_llm.ainvoke = AsyncMock(return_value=enrich_resp)
        mock_enrich_llm.return_value = enrich_llm

        # -- Amadeus mock --
        mock_search_flights.side_effect = lambda *a, **kw: [_mock_flight_offer("TST")]

        # -- Weather mock --
        mock_get_weather.return_value = WeatherInfo(avg_temp_c=28.0, description="Warm")

        # -- Report mock --
        report_path = str(tmp_path / "provider_test_report.pdf")
        mock_generate_report.return_value = report_path

        # -- Store interaction: no user_id --
        mock_store_embed_texts.return_value = [_random_vector(seed=99)]

        from src.app.graph.pipeline import build_graph

        graph = build_graph()
        initial_state = {
            "user_message": "Beach trip from Bogota",
            "user_id": "",
            "validated": False,
            "validation_errors": [],
            "candidate_destinations": [],
            "destination_reports": [],
            "enriched": False,
            "report_path": "",
            "suggest_retry_count": 0,
            "errors": [],
        }

        result = await graph.ainvoke(initial_state)

        # Pipeline completed
        assert result.get("report_path") == report_path

        # All three get_llm mocks were called -- proving nodes use the factory
        mock_intake_llm.assert_called_once_with(temperature=0.3)
        mock_suggest_llm.assert_called_once_with(temperature=0.7)
        # Enrich get_llm is called once (for activities generation)
        mock_enrich_llm.assert_called_once_with(temperature=0.7)


class TestRealQdrantSearchInPipeline:
    """Tests that use real Qdrant embedded search (not mocked) to verify
    the RAG query functions work end-to-end with actual vector similarity."""

    @patch("src.app.services.rag.embed_query")
    async def test_query_travel_knowledge_returns_seeded_data(
        self,
        mock_embed_query,
        seeded_qdrant,
    ):
        """Verify that query_travel_knowledge with a real Qdrant returns
        results from the seeded travel_knowledge collection."""
        # Return a vector similar to VEC_BEACH (seed=42)
        mock_embed_query.return_value = VEC_QUERY

        from src.app.services.rag import query_travel_knowledge

        results = await query_travel_knowledge("beach tropical", limit=3)

        assert len(results) > 0
        # The closest result should be the CUN beach chunk (same vector seed)
        top_result = results[0]
        assert top_result["payload"]["city"] == "Cancun"
        assert top_result["score"] > 0.5

    @patch("src.app.services.rag.embed_query")
    async def test_query_travel_knowledge_with_iata_filter(
        self,
        mock_embed_query,
        seeded_qdrant,
    ):
        """Verify that IATA filtering scopes results to a single destination."""
        mock_embed_query.return_value = VEC_QUERY

        from src.app.services.rag import query_travel_knowledge

        # Filter to BCN only
        results = await query_travel_knowledge(
            "culture architecture",
            limit=5,
            destination_iata="BCN",
        )

        # All results should be BCN
        for r in results:
            assert r["payload"]["iata"] == "BCN"

    @patch("src.app.services.rag.embed_query")
    async def test_query_interactions_empty_without_user_id(
        self,
        mock_embed_query,
        seeded_qdrant,
    ):
        """query_interactions with empty user_id returns empty list
        without even calling embed_query."""
        from src.app.services.rag import query_interactions

        results = await query_interactions("some query", user_id="", limit=3)

        assert results == []
        mock_embed_query.assert_not_called()
