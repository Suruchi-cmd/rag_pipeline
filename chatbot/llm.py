"""
Local Ollama LLM client (llama3.1:8b).

Uses the OpenAI-compatible API exposed by Ollama (host configurable via
OLLAMA_BASE_URL — must end with /v1).
Streaming via SSE for chat_handler, async streaming for voice_handler.

Key design decisions:
- One sync OpenAI client for chat_handler (threading bridge).
- One AsyncOpenAI client for voice_handler (native async streaming).
- 3 attempts with exponential backoff on transient errors.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Generator

from openai import AsyncOpenAI, OpenAI

from chatbot.config import settings

logger = logging.getLogger(__name__)

_FALLBACK_MSG = settings.fallback_message
_OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
_OLLAMA_REPHRASE_URL = settings.OLLAMA_REPHRASE_URL
_MODEL = settings.LLM_MODEL


_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None
_rephrase_client: AsyncOpenAI | None = None


def _make_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")
    return _client


def _make_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")
    return _async_client


def _make_rephrase_client() -> AsyncOpenAI:
    global _rephrase_client
    if _rephrase_client is None:
        _rephrase_client = AsyncOpenAI(base_url=_OLLAMA_REPHRASE_URL, api_key="ollama")
    return _rephrase_client


def generate_response(messages: list[dict]) -> Generator[str, None, None]:
    """
    Streaming token generator.

    Yields str tokens as they arrive from the Ollama API.
    Retries up to 3 times on transient errors with exponential backoff.
    Raises on unrecoverable errors so callers can substitute the fallback message.
    """
    delay = 1.0

    for attempt in range(settings.LLM_RETRIES):
        try:
            client = _make_client()
            stream = client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                stream=True,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    yield token
            return  # success — stop retry loop

        except Exception as exc:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", 0) if resp else 0
            if status in (429, 503) and attempt < settings.LLM_RETRIES - 1:
                logger.warning(
                    "Ollama rate-limited (HTTP %s), retry %d/%d in %.1fs",
                    status,
                    attempt + 1,
                    settings.LLM_RETRIES - 1,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error("Ollama API error (attempt %d): %s", attempt + 1, exc)
                raise


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>…</think> reasoning blocks from output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def generate_response_sync(messages: list[dict]) -> str:
    """Non-streaming convenience wrapper — collects the full response into a string."""
    return strip_thinking("".join(generate_response(messages)))
