"""
Local Ollama async client (OpenAI-compatible API surface).

The voice pipeline streams via this AsyncOpenAI client; per-token cleaning
and <think>-stripping live in voice_handler.clean_for_tts.

OLLAMA_BASE_URL must end with /v1.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from chatbot.config import settings

_FALLBACK_MSG = settings.fallback_message
_OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL

_async_client: AsyncOpenAI | None = None


def _make_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")
    return _async_client


async def close_llm_client() -> None:
    """Close the shared async client. Call from FastAPI lifespan shutdown."""
    global _async_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None
