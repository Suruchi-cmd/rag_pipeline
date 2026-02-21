"""
Voice call handler for Twilio integration.

Mirrors chat_handler.py but adapted for phone calls:
- Non-streaming LLM call (Twilio needs complete text, not SSE tokens)
- Voice-specific system prompt (2 sentences, no markdown)
- Uses CallSid as session_id so history is shared with the conversation store
- max_tokens=150 for faster, concise TTS responses

Actual API shapes (verified against existing code):
  search:       hybrid_search(query, category, top_k)  / semantic_search(...)
  conversation: conversation_store.get(session_id) → list[dict]
                conversation_store.add(session_id, role, content)
  llm:          _make_client() + client.chat.completions.create(stream=False)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys

# Same sys.path guard used by chat_handler.py — lets us import flat RAG modules.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from search import hybrid_search, semantic_search  # noqa: E402

from chatbot.conversation import conversation_store  # noqa: E402
from chatbot.llm import _make_client  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice-specific system prompt (replaces the web-chat SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

VOICE_SYSTEM_PROMPT = (
    "You are AeroBot, the friendly voice assistant for AeroSports Scarborough. "
    "Answer the caller's question clearly in two sentences or fewer. "
    "Do not use bullet points, markdown, emojis, or lists. "
    "Speak naturally as if talking on the phone. "
    "All prices are plus tax. "
    "If they want to book or need more detail, direct them to call 289-454-5555 "
    "or visit aerosportsparks.ca."
)

_VOICE_FALLBACK = (
    "I'm having trouble answering right now. "
    "Please call us directly at 289-454-5555."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_for_tts(text: str) -> str:
    """Strip markdown and symbols that sound bad when read aloud by TTS."""
    text = re.sub(r"\*+", "", text)                               # bold / italic asterisks
    text = re.sub(r"#+\s*", "", text)                             # ATX headings
    text = re.sub(r"`+", "", text)                                # inline code / fences
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)        # [label](url) → label
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)  # leading bullet dashes
    text = re.sub(r"\n+", " ", text).strip()
    return text


def _build_voice_messages(
    user_message: str,
    rag_context: list,
    conversation_history: list[dict],
) -> list[dict]:
    """
    Assemble the messages list for the LLM using the voice system prompt.

    Structure mirrors prompt_builder.build_messages() but with VOICE_SYSTEM_PROMPT
    so we don't need to modify the shared prompt_builder module.
    """
    if rag_context:
        lines = ["KNOWLEDGE BASE CONTEXT:\n"]
        for i, result in enumerate(rag_context, 1):
            c = result.chunk
            lines.append(f"[{i}] {c.category} > {c.subcategory}")
            lines.append(f"Q: {c.question}")
            lines.append(f"A: {c.answer}")
            lines.append("")
        context_text = "\n".join(lines)
    else:
        context_text = (
            "KNOWLEDGE BASE CONTEXT:\n\n"
            "No matching context was found for this query. "
            "Direct the caller to phone 289-454-5555 or email events.scb@aerosportsparks.ca."
        )

    messages: list[dict] = [
        {"role": "system", "content": VOICE_SYSTEM_PROMPT},
        {"role": "system", "content": context_text},
    ]
    # Reuse existing history — keep last 10 turns (20 messages) max
    messages.extend(conversation_history[-20:])
    messages.append({"role": "user", "content": user_message})
    return messages


def _call_llm_sync(messages: list[dict]) -> str:
    """Blocking non-streaming LLM call with a short token budget for voice."""
    model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    client = _make_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=150,
        temperature=0.3,
        top_p=0.9,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def handle_voice_message(call_sid: str, user_text: str) -> str:
    """
    Main entry point for Twilio voice calls.

    call_sid is used as the session_id so this call's history lives alongside
    any web-chat sessions in the shared conversation_store.

    Returns a TTS-clean string ready for Twilio's <Say> verb.
    """
    user_text = user_text.strip()[:500]

    # RAG retrieval — same concurrent pattern as chat_handler.py
    try:
        semantic_results, hybrid_results = await asyncio.gather(
            asyncio.to_thread(semantic_search, user_text, None, 5),
            asyncio.to_thread(hybrid_search, user_text, None, 3),
        )
        seen: set[str] = set()
        merged = []
        for r in semantic_results + hybrid_results:
            if r.chunk.id not in seen:
                seen.add(r.chunk.id)
                merged.append(r)
            if len(merged) >= 5:
                break
    except Exception as exc:
        logger.error("Search failed for voice call %s: %s", call_sid, exc)
        merged = []

    # Build voice-specific prompt
    history = conversation_store.get(call_sid)
    messages = _build_voice_messages(user_text, merged, history)

    # Non-streaming LLM call — run blocking I/O in a thread
    try:
        reply = await asyncio.to_thread(_call_llm_sync, messages)
    except Exception as exc:
        logger.error("LLM error on voice call %s: %s", call_sid, exc)
        reply = _VOICE_FALLBACK

    reply = _clean_for_tts(reply)

    # Persist turn to the shared store (same TTL / trim logic applies)
    conversation_store.add(call_sid, "user", user_text)
    conversation_store.add(call_sid, "assistant", reply)

    return reply
