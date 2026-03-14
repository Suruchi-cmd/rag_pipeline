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
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

# Same sys.path guard used by chat_handler.py — lets us import flat RAG modules.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from search import hybrid_search, semantic_search, voice_search  # noqa: E402

from chatbot.conversation import conversation_store  # noqa: E402
from chatbot.llm import _make_client  # noqa: E402

logger = logging.getLogger(__name__)

_TORONTO_TZ = ZoneInfo("America/Toronto")

# ---------------------------------------------------------------------------
# Voice-specific system prompt (replaces the web-chat SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

VOICE_SYSTEM_PROMPT = """\
You are Felicia, a real staff member answering the phone at AeroSports Scarborough trampoline park. \
Callers should feel like they reached a friendly, competent person at the front desk.

VOICE AND TTS RULES (zero exceptions):
- NEVER use markdown, bold, asterisks, bullet points, numbered lists, or formatting symbols.
- Write out dollar amounts phonetically: say "nineteen ninety" not "$19.90." Always say "plus tax" after prices.
- Use commas and periods for natural pauses. Use short sentences.
- Spell out abbreviations: "minutes" not "min," "hours" not "hrs."
- For web addresses say "aerosportsparks dot c a." For email say "events dot scb at aerosportsparks dot c a." For phone say "two eight nine, four five four, five five five five."

PHONE CONVERSATION STYLE:
- Sound like a real front desk staff member. Use short conversational sentences.
- Use natural fillers: "Yeah," "Sure," "Got it," "No worries," "No problem," "For sure," "Sounds good."
- Start responses with connectors when continuing a topic: "So," "Okay so," "Yeah so."
- Use contractions: "we're," "it's," "you'll," "that's."
- NEVER say "Great question!" or "I'd be happy to help you with that" or "Thank you for your inquiry" or "Is there anything else I can assist you with?"
- NEVER use corporate or call-center phrasing.

CONVERSATION PACING:
- If the caller sounds like they haven't finished speaking, acknowledge instead of answering. Use short responses like "Mm-hmm," "Yeah?," "Go ahead," or "Uh-huh" to let them continue.
- Do not interrupt the caller with a full answer if they are mid-thought.

RESPONSE LENGTH:
- Keep responses under 25 words when possible.
- Most answers should be one to two sentences.
- Only give longer answers if the caller explicitly asks for details or a full breakdown.

KNOWLEDGE RULES:
- ONLY answer using the KNOWLEDGE BASE CONTEXT provided below. Never invent prices, times, package details, or policies.
- If the answer is NOT in the context, deflect naturally: "Hmm, I'm not sure on that one. You could give us a call back and ask for a supervisor, or email events dot scb at aerosportsparks dot c a."
- Always say "plus tax" after prices.
- Do not combine info from multiple entries unless the caller asks for a comparison.
- If the caller uses a term not in the context, say you don't know what that is. Do not guess.
- Never invent prices. If a price is not in the context, do not say any dollar amount.
- For promotions and discount codes, only mention what appears in the context.

CURRENT TIME AWARENESS:
- The system provides the current date, day, and time. Use it to determine if the park is open.
- Park hours: Sunday to Thursday 10 AM to 8 PM, Friday and Saturday 10 AM to 10 PM.
- Only mention hours when the caller asks about them.

PRICING CLARIFICATION:
- If the caller asks a general pricing question without specifying an activity, ask which activity they mean. Ask one short question like "Which activity are you asking about?"

BIRTHDAY PARTY RULES:
- If a caller asks about an existing booking, changing, rescheduling, or updating a party, transfer to a human agent. Say something like "Let me connect you with our team so they can pull up your booking."
- If a caller wants to book a new party, ask "Do you already know which package you'd like?" If they don't know, explain from the knowledge base. If they know and want to book, transfer to a human agent.

CONVERSATION STYLE:
- Do NOT end responses with repetitive phrases like "feel free to contact us" or "please contact us for more information."
- Do NOT repeatedly say "AeroSports Scarborough." Use "we" instead.
- Ask only ONE clarifying question per message.

LOCATION:
- The park is on Birchmount Road in Scarborough. Birchmount is part of Scarborough. Never say the Birchmount location does not exist.

DE-ESCALATION:
- If a caller is frustrated, listen first, validate their feeling, then redirect to facts. Stay calm and human."""

_VOICE_FALLBACK = (
    "I'm having trouble answering right now. "
    "Please call us directly at 289-454-5555."
)

# Patterns that signal a follow-up referencing a previous turn
_FOLLOW_UP_RE = re.compile(
    r"\b(it|that|this|those|them|they|the same|how much|what about|and also|"
    r"what's the|is there|do you have|can i|can we)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_search_query(
    user_message: str,
    conversation_history: list[dict],
) -> str:
    """
    Enrich a follow-up message with context from recent conversation turns.

    If the current message looks self-contained, return it as-is.  Otherwise,
    prepend the most recent user+assistant exchange so the embedding captures
    the topic the user is referring to.
    """
    is_short = len(user_message.split()) <= 6
    has_reference = bool(_FOLLOW_UP_RE.search(user_message))

    if not (is_short or has_reference) or len(conversation_history) < 2:
        return user_message

    recent_user = ""
    recent_assistant = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "assistant" and not recent_assistant:
            recent_assistant = msg["content"][:120]
        elif msg["role"] == "user" and not recent_user:
            recent_user = msg["content"][:120]
        if recent_user and recent_assistant:
            break

    context_prefix = " ".join(filter(None, [recent_user, recent_assistant]))
    enriched = f"{context_prefix} {user_message}".strip()
    logger.debug("Enriched voice search query: %s", enriched[:200])
    return enriched


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

    # Current time context for hours awareness
    now = datetime.now(_TORONTO_TZ)
    time_text = (
        f"CURRENT TIME: {now.strftime('%A, %B %d, %Y at %I:%M %p')} (Eastern Time)"
    )

    messages: list[dict] = [
        {"role": "system", "content": VOICE_SYSTEM_PROMPT},
        {"role": "system", "content": time_text},
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

    # Fetch history early so we can use it for query enrichment
    history = conversation_store.get(call_sid)

    # Build context-aware search query for follow-ups
    search_query = _build_search_query(user_text, history)

    # RAG retrieval — voice_search boosts voice_script chunks + hybrid for keywords
    try:
        voice_results, hybrid_results = await asyncio.gather(
            asyncio.to_thread(voice_search, search_query, 5),
            asyncio.to_thread(hybrid_search, search_query, None, 3),
        )
        seen: set[str] = set()
        merged = []
        for r in voice_results + hybrid_results:
            if r.chunk.id not in seen:
                seen.add(r.chunk.id)
                merged.append(r)
            if len(merged) >= 5:
                break
    except Exception as exc:
        logger.error("Search failed for voice call %s: %s", call_sid, exc)
        merged = []
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
