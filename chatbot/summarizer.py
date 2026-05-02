"""Generate a short call summary for the follow-up email body.

Uses `OllamaChat` from src/ai/localllm.py with the same `VOICE_FAST_MODEL`
configured for the voice pipeline. The chat instance is cached so we don't
rebuild the LangChain engines and prompt templates on every call.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from chatbot.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You write concise call summaries for AeroSports Scarborough's events team. "
    "Given a phone-call transcript between a Caller and Maya (the AI agent), "
    "produce a single short paragraph (3–5 sentences max) that captures: "
    "(1) what the caller wanted, (2) any key details mentioned (names, dates, "
    "party size, package interest, phone callback), and (3) what action is "
    "needed from the team. Plain text only — no markdown, no preamble like "
    "'Summary:'. If the transcript is empty or unintelligible, reply exactly: "
    "'No usable transcript captured.'"
)

_FALLBACK_SUMMARY = "No usable transcript captured."


@lru_cache(maxsize=1)
def _get_chat():
    from src.ai.localllm import OllamaChat

    return OllamaChat(
        model=settings.VOICE_FAST_MODEL,
        base_url=settings.ollama_embed_url,
        system_prompt=_SYSTEM_PROMPT,
        temperature=0.2,
        num_predict=300,
    )


def _format_transcript(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        speaker = "Caller" if m.get("role") == "user" else "Maya"
        lines.append(f"{speaker}: {content[:500]}")
    return "\n".join(lines[:80])


async def summarize_call(messages: list[dict]) -> str:
    """Return a short LLM-generated summary of the transcript."""
    transcript = _format_transcript(messages)
    if not transcript:
        return _FALLBACK_SUMMARY

    try:
        chat = _get_chat()
        result = await chat.achat(f"Transcript:\n{transcript}")
        text = (result or "").strip()
        return text or _FALLBACK_SUMMARY
    except Exception as exc:
        logger.error("Call summarization failed: %s", exc)
        return _FALLBACK_SUMMARY
