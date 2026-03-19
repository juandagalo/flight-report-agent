"""Factory function for LLM provider abstraction."""

from langchain_core.language_models.chat_models import BaseChatModel

from src.app.config import settings


def get_llm(temperature: float = 0.7) -> BaseChatModel:
    """Return a BaseChatModel instance based on LLM_PROVIDER config.

    Supports:
      - "openai"  -> ChatOpenAI (default)
      - "claude"  -> ChatAnthropic

    Raises ValueError for unknown providers.
    """
    provider = settings.LLM_PROVIDER.lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
        )
    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=temperature,
            api_key=settings.ANTHROPIC_API_KEY,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Use 'openai' or 'claude'."
        )
