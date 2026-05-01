"""
Local Ollama LLM client.

Uses the OpenAI-compatible API exposed by Ollama (hosts configurable via
OLLAMA_URL_1 / OLLAMA_URL_2 — both must end with /v1).

Two Ollama instances are kept in a round-robin pool, but assignment is
*per call session*, not per request:

- get_session_client(call_sid) lazily pins the next round-robin client to
  that call_sid. Every subsequent turn in the same call hits the same
  Ollama instance. Two concurrent calls naturally land on different
  instances; an Nth concurrent call wraps around the cycle.
- swap_session_client(call_sid) re-pins to the other instance on failover.
- release_session_client(call_sid) drops the pin when the call ends.

Key design decisions:
- One sync OpenAI client (URL_1) for chat_handler threading bridge.
- Async pool pinned per call_sid for voice_handler streaming + rewrite +
  mid-call classifier.
- _make_async_client() is kept for one-shot, session-less work
  (post-call classification): it round-robins per request.
- 3 attempts with exponential backoff on transient errors.
"""

from __future__ import annotations

import itertools
import logging
import re
import threading
import time
from typing import Generator

from openai import AsyncOpenAI, OpenAI

from chatbot.config import settings

logger = logging.getLogger(__name__)

_FALLBACK_MSG = settings.fallback_message
_MODEL = settings.LLM_MODEL

# ── Sync client (chat_handler only) ────────────────────────────────────────

_client: OpenAI | None = None


def _make_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=settings.OLLAMA_URL_1, api_key="ollama")
    return _client


# ── Async round-robin pool ──────────────────────────────────────────────────

_pool: list[AsyncOpenAI] = []
_cycle: itertools.cycle | None = None
_pool_lock = threading.Lock()
_session_clients: dict[str, AsyncOpenAI] = {}


def _init_pool() -> None:
    global _pool, _cycle
    _pool = [
        AsyncOpenAI(base_url=settings.OLLAMA_URL_1, api_key="ollama"),
        AsyncOpenAI(base_url=settings.OLLAMA_URL_2, api_key="ollama"),
    ]
    _cycle = itertools.cycle(_pool)


def _make_async_client() -> AsyncOpenAI:
    """Return next client in round-robin pool. Session-less callers only."""
    with _pool_lock:
        if not _pool:
            _init_pool()
        return next(_cycle)  # type: ignore[arg-type]


def _get_fallback_async_client(current: AsyncOpenAI) -> AsyncOpenAI:
    """Return the other client in the pool (fallback when current fails)."""
    with _pool_lock:
        for c in _pool:
            if c is not current:
                return c
    return current


def get_session_client(call_sid: str) -> AsyncOpenAI:
    """Return the Ollama client pinned to this call session.

    Lazily assigned on first use from the round-robin pool, then held for
    the lifetime of the call so every turn lands on the same instance.
    """
    with _pool_lock:
        if not _pool:
            _init_pool()
        client = _session_clients.get(call_sid)
        if client is None:
            client = next(_cycle)  # type: ignore[arg-type]
            _session_clients[call_sid] = client
            logger.info(
                "[%s] Pinned Ollama client → %s", call_sid, client.base_url
            )
        return client


def swap_session_client(call_sid: str) -> AsyncOpenAI:
    """Re-pin the session to the other pool client and return it.

    Used when the currently-pinned client raises a transient error so the
    rest of the call lands on the healthy instance.
    """
    with _pool_lock:
        if not _pool:
            _init_pool()
        current = _session_clients.get(call_sid)
        for c in _pool:
            if c is not current:
                _session_clients[call_sid] = c
                logger.info(
                    "[%s] Swapped Ollama client → %s", call_sid, c.base_url
                )
                return c
        return current  # type: ignore[return-value]


def release_session_client(call_sid: str) -> None:
    """Drop the session→client pin when the call ends."""
    with _pool_lock:
        _session_clients.pop(call_sid, None)


# ── Sync streaming generator (chat_handler) ────────────────────────────────


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
