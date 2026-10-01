import logging

from app.core.config import settings
from app.llm.cache import CachedProvider
from app.llm.groq_provider import GroqProvider
from app.llm.mock import MockProvider

logger = logging.getLogger("app.llm")


def get_provider():
    if settings.llm_provider == "mock":
        provider = MockProvider()
    elif settings.llm_provider == "groq":
        provider = CachedProvider(
            GroqProvider(settings.groq_api_key, settings.groq_model, settings.llm_timeout_s),
            settings.llm_cache_dir,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
    logger.info(
        "llm_provider_selected",
        extra={"extra_fields": {"provider": provider.name, "model": provider.model}},
    )
    return provider