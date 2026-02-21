"""
HuggingFace Inference Provider client for Llama.

Uses huggingface_hub.InferenceClient with the OpenAI-compatible chat-completions
API, streaming via SSE.

Key design decisions:
- One fresh InferenceClient per call (lightweight, avoids stale state).
- Sync generator for tokens — chat_handler.py bridges to async via threading.
- 3 attempts with exponential backoff on 429 / 503.
- HF_PROVIDER env var is optional; empty string → auto-routing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Generator

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

_FALLBACK_MSG = (
    "I'm having trouble connecting right now. "
    "Please try again in a moment, or call us at **289-454-5555** for immediate help!"
)


def _make_client() -> InferenceClient:
    token = os.environ.get("HF_TOKEN")
    provider = os.environ.get("HF_PROVIDER") or None  # "" → None → auto-route
    return InferenceClient(api_key=token, provider=provider)


def generate_response(messages: list[dict]) -> Generator[str, None, None]:
    """
    Streaming token generator.

    Yields str tokens as they arrive from the HuggingFace API.
    Retries up to 3 times on 429 / 503 with exponential backoff.
    Raises on unrecoverable errors so callers can substitute the fallback message.
    """
    model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    delay = 1.0

    for attempt in range(3):
        try:
            client = _make_client()
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=1024,
                temperature=0.3,
                top_p=0.9,
            )
            for chunk in stream:
                # Some HF stream frames arrive with an empty choices list
                # (e.g. usage/stats frames at the end of the stream).
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
                    "HuggingFace rate-limited (HTTP %s), retry %d/2 in %.1fs",
                    status,
                    attempt + 1,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error("HuggingFace API error (attempt %d): %s", attempt + 1, exc)
                raise


def generate_response_sync(messages: list[dict]) -> str:
    """Non-streaming convenience wrapper — collects the full response into a string."""
    return "".join(generate_response(messages))
