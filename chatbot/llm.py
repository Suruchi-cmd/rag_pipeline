"""
Local Ollama LLM client (llama3.1:8b).

Uses the OpenAI-compatible API exposed by Ollama at localhost:11434.
Streaming via SSE for chat_handler, async streaming for voice_handler.

Key design decisions:
- One sync OpenAI client for chat_handler (threading bridge).
- One AsyncOpenAI client for voice_handler (native async streaming).
- 3 attempts with exponential backoff on transient errors.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Generator

from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

_FALLBACK_MSG = (
    "I'm so sorry, I'm having a little trouble on my end right now. "
    "Can you try calling back in just a minute? Or you can reach us "
    "at two eight nine, four five four, five five five five. Sorry about that!"
)

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.50.150:11434/v1")
_MODEL = os.environ.get("LLM_MODEL", "phi4:latest")


_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


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


def generate_response(messages: list[dict]) -> Generator[str, None, None]:
    """
    Streaming token generator.

    Yields str tokens as they arrive from the Ollama API.
    Retries up to 3 times on transient errors with exponential backoff.
    Raises on unrecoverable errors so callers can substitute the fallback message.
    """
    delay = 1.0

    for attempt in range(3):
        try:
            client = _make_client()
            stream = client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                stream=True,
                max_tokens=1024,
                temperature=0.3,
                top_p=0.9,
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
            if status in (429, 503) and attempt < 2:
                logger.warning(
                    "Ollama rate-limited (HTTP %s), retry %d/2 in %.1fs",
                    status,
                    attempt + 1,
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
