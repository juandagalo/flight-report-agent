"""Provider-agnostic embedding generation service."""

from langchain_core.embeddings import Embeddings

from src.app.config import settings

_embeddings: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """Return a cached Embeddings singleton.

    Provider mapping:
      - "openai"  -> OpenAIEmbeddings (text-embedding-3-small)
      - "claude"  -> OpenAIEmbeddings (text-embedding-3-small)
        Note: Anthropic does not offer a native embedding model.
        We always use OpenAI embeddings regardless of LLM provider.
        This keeps vector dimensions consistent across provider switches.

    Requires OPENAI_API_KEY to be set.
    """
    global _embeddings
    if _embeddings is None:
        from langchain_openai import OpenAIEmbeddings

        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )
    return _embeddings


def reset_embeddings() -> None:
    """Reset the cached embeddings instance. Used in tests."""
    global _embeddings
    _embeddings = None


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings.

    Returns a list of float vectors, one per input text.
    """
    embeddings = get_embeddings()
    return await embeddings.aembed_documents(texts)


async def embed_query(text: str) -> list[float]:
    """Generate an embedding for a single query string.

    Uses the query-optimized embedding method when available.
    """
    embeddings = get_embeddings()
    return await embeddings.aembed_query(text)
