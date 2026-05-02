"""
Voice call handler for Twilio inbound calls.

Pipeline per turn:
  1. Rewrite the caller utterance into a standalone search query (LLM).
  2. Run vector search directly against pgvector (no external RAG service).
  3. Stream an LLM reply back, chunked at sentence boundaries for TTS.

The booking-capture state machine and end-of-call classifier live alongside
the streaming entry points because the WebSocket handler in server.py drives
both flows from the same import surface.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# Allow flat-module imports (src.utils, etc.) when run from the chatbot package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chatbot.config import settings  # noqa: E402
from chatbot.conversation import conversation_store  # noqa: E402
from chatbot.llm import get_session_client, swap_session_client  # noqa: E402
from chatbot.prompt_defaults import (  # noqa: E402
    CLASSIFIER_PROMPT_SLUG,
    REWRITE_PROMPT_SLUG,
    VOICE_SYSTEM_PROMPT_SLUG,
)
from chatbot.prompt_loader import get_prompt  # noqa: E402
from chatbot.vector_store import vector_store  # noqa: E402
from src.utils.pipeline_logger import PipelineLogger  # noqa: E402

logger = logging.getLogger(__name__)

_TZ = ZoneInfo(settings.TIMEZONE)
_VOICE_MODEL = settings.VOICE_LLM_MODEL
_FAST_MODEL = settings.VOICE_FAST_MODEL


# ---------------------------------------------------------------------------
# Per-session pipeline loggers
# ---------------------------------------------------------------------------

_session_loggers: dict[str, PipelineLogger] = {}


def init_session_logger(call_sid: str) -> PipelineLogger:
    pl = PipelineLogger(call_sid)
    _session_loggers[call_sid] = pl
    return pl


def get_session_logger(call_sid: str) -> PipelineLogger | None:
    return _session_loggers.get(call_sid)


def close_session_logger(call_sid: str) -> None:
    pl = _session_loggers.pop(call_sid, None)
    if pl is not None:
        pl.close()


# ---------------------------------------------------------------------------
# System prompt — content lives in the database (chatbot.prompt_loader).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# End-of-call detection
# ---------------------------------------------------------------------------

_END_CALL_KEYWORDS_DEFINITE = [
    "bye",
    "goodbye",
    "good bye",
    "that's all",
    "thats all",
    "thanks bye",
    "thank you bye",
    "no thanks bye",
    "i'm good thanks",
    "im good thanks",
    "that's everything",
    "thats everything",
    "i'll call back",
    "ill call back",
    "i'll call later",
    "ill call later",
    "have a good one",
    "you too bye",
]

_END_CALL_KEYWORDS_MAYBE = [
    "speak to a manager",
    "speak to manager",
    "talk to a manager",
    "talk to manager",
    "speak to someone",
    "speak to a supervisor",
    "talk to a supervisor",
    "human agent",
    "real person",
    "actual person",
    "a person",
    "my booking",
    "my reservation",
    "my party",
    "i already booked",
    "i booked",
    "my existing booking",
    "change my booking",
    "cancel my booking",
    "reschedule",
    "refund",
    "complaint",
    "i want to complain",
    "file a complaint",
    "unhappy with",
    "not happy with",
]

# High-confidence triggers that the caller wants to modify an existing booking.
# server.py uses these to bypass the LLM and start a name/phone/details capture.
_BOOKING_CAPTURE_TRIGGERS = [
    "change my booking",
    "change my party",
    "change my reservation",
    "cancel my booking",
    "cancel my party",
    "cancel my reservation",
    "reschedule my",
    "modify my booking",
    "modify my party",
    "update my booking",
    "update my party",
    "my existing booking",
    "change the booking",
    "change the time of my",
    "move my party",
    "move my booking",
    "change my kid's party",
    "change my daughter's party",
    "change my son's party",
    "change my child's party",
    "i booked a party",
    "i already booked",
    "we already booked",
    "change the time",
    "change time for",
    "change time of",
    "move the time",
    "i have a booking",
    "i have a party",
    "i have my booking",
    "i have my party",
    "i have my birthday",
    "i have my birth",  # ASR often drops "day"
    "booking tomorrow",
    "party tomorrow",
    "booked for tomorrow",
    "my birthday party",
    "my birthday booking",
]

# Pure acknowledgments / greetings — answerable from history alone, no RAG.
_SKIP_RAG_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|okay|ok|no|nope|nah|thanks|thank you|"
    r"bye|goodbye|hi|hello|hey|good morning|good afternoon|good evening|"
    r"mm.?hmm|uh.?huh|go ahead|please|right|got it|sounds good|awesome|"
    r"cool|great|perfect|alright|nice|for sure|no worries)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def check_end_keywords(user_text: str) -> str:
    """Return 'definite' | 'maybe' | 'none' for the user's last utterance."""
    lowered = user_text.lower().strip()
    if not lowered:
        return "none"
    for kw in _END_CALL_KEYWORDS_DEFINITE:
        if kw in lowered:
            return "definite"
    for kw in _END_CALL_KEYWORDS_MAYBE:
        if kw in lowered:
            return "maybe"
    return "none"


def check_booking_capture_trigger(user_text: str) -> bool:
    lowered = user_text.lower().strip()
    if not lowered:
        return False
    return any(trigger in lowered for trigger in _BOOKING_CAPTURE_TRIGGERS)


async def classify_turn_for_end(
    call_sid: str, user_text: str, assistant_text: str
) -> dict | None:
    """Return an end-call decision dict, or None to keep the call going."""
    template = get_prompt(CLASSIFIER_PROMPT_SLUG)
    prompt = template.replace("<<USER_TEXT>>", user_text[:500]).replace(
        "<<ASSISTANT_TEXT>>", assistant_text[:500]
    )

    try:
        client = get_session_client(call_sid)
        response = await client.chat.completions.create(
            model=_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=settings.REWRITE_MAX_TOKENS,
            temperature=settings.REWRITE_TEMPERATURE,
        )
        raw = (response.choices[0].message.content or "").strip()
        logger.info("Classifier raw output: %r", raw[:300])

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("Classifier returned no JSON object")
            return None

        parsed = json.loads(match.group(0))
        if not parsed.get("should_end"):
            return None

        return {
            "summary": parsed.get("summary", "Call ended"),
            "needs_human": bool(parsed.get("needs_human", False)),
            "flag_reason": parsed.get("flag_reason", "") or "",
        }
    except Exception as exc:
        logger.error("Classifier failed, not ending call: %s", exc)
        return None


def build_end_decision_from_definite(user_text: str, assistant_text: str) -> dict:
    """Cheap end-decision when a 'definite' keyword fired — skip the LLM call."""
    return {
        "summary": f"Caller ended the call after: {user_text[:100]}",
        "needs_human": False,
        "flag_reason": "",
    }


# ---------------------------------------------------------------------------
# Query rewriting + RAG-skip heuristic
# ---------------------------------------------------------------------------


def _should_skip_rag(user_message: str, conversation_history: list[dict]) -> bool:
    """Skip RAG only when the message is a pure ack AND we have history."""
    if not conversation_history:
        return False
    return bool(_SKIP_RAG_RE.match(user_message))


async def _rewrite_query(
    call_sid: str, user_message: str, conversation_history: list[dict]
) -> str:
    """Rewrite a follow-up into a self-contained query. Falls back on error."""
    if not conversation_history:
        return user_message

    recent = conversation_history[-settings.REWRITE_HISTORY_TURNS :]
    history_lines = [
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in recent
    ]
    chat_history = "\n".join(history_lines)
    template = get_prompt(REWRITE_PROMPT_SLUG)
    prompt = template.format(chat_history=chat_history, question=user_message)

    try:
        client = get_session_client(call_sid)
        response = await client.chat.completions.create(
            model=_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=settings.REWRITE_MAX_TOKENS,
            temperature=settings.REWRITE_TEMPERATURE,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        if not rewritten or len(rewritten) > 500:
            return user_message
        logger.info("Rewritten query: %r → %r", user_message[:100], rewritten[:200])
        return rewritten
    except Exception as exc:
        logger.error("Query rewrite failed, using original: %s", exc)
        return user_message


# ---------------------------------------------------------------------------
# TTS cleaning
# ---------------------------------------------------------------------------


def _clean_token_for_tts(token: str) -> str:
    """Per-token cleaner — strip markdown symbols, keep whitespace."""
    token = re.sub(r"\*+", "", token)
    token = re.sub(r"#+", "", token)
    token = re.sub(r"`+", "", token)
    token = re.sub(r"\$", "", token)
    return token


def clean_for_tts(text: str) -> str:
    """Full cleaner for accumulated text written to conversation history."""
    # Strip Qwen3-style <think>…</think> reasoning blocks (closed, unclosed, orphaned).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    # TTS reads emojis as their text description ("party popper") — strip them.
    text = re.sub(r"[\U00010000-\U0010FFFF]", "", text)
    text = re.sub(r"[☀-➿︀-️]", "", text)
    text = re.sub(r"\bmins?\b", "minutes", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhrs?\b", "hours", text, flags=re.IGNORECASE)
    # "$39.90" / "$39" / bare "39.90" → spoken form "39 90" / "39 dollars".
    text = re.sub(r"\$(\d+)\.(\d{2})", r"\1 \2", text)
    text = re.sub(r"\$(\d+)", r"\1 dollars", text)
    text = re.sub(r"\b(\d+)\.(\d{2})\b", r"\1 \2", text)
    text = re.sub(r"\n+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------


def _build_voice_messages(
    user_message: str,
    rag_context: list,
    conversation_history: list[dict],
) -> list[dict]:
    if rag_context:
        # RAG often returns the same content from different source paths — dedupe.
        seen_content: set[str] = set()
        unique_docs: list[dict] = []
        for doc in rag_context:
            content = doc.get("content", "").strip()
            if content and content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)

        lines = ["KNOWLEDGE BASE CONTEXT:\n"]
        for i, doc in enumerate(unique_docs, 1):
            source = doc.get("metadata", {}).get("file_name", "unknown")
            lines.append(f"[{i}] (source: {source})")
            lines.append(doc.get("content", ""))
            lines.append("")
        context_text = "\n".join(lines)
    else:
        context_text = (
            "KNOWLEDGE BASE CONTEXT:\n\n"
            "No matching context was found for this query. "
            f"Direct the caller to phone {settings.BUSINESS_PHONE} "
            f"or email {settings.BUSINESS_EMAIL}."
        )

    now = datetime.now(_TZ)
    time_text = (
        f"CURRENT TIME: {now.strftime('%A, %B %d, %Y at %I:%M %p')} (Eastern Time)"
    )

    system_prompt = get_prompt(VOICE_SYSTEM_PROMPT_SLUG)
    system_content = f"{system_prompt}\n\n{time_text}\n\n{context_text}"
    messages: list[dict] = [{"role": "system", "content": system_content}]
    messages.extend(conversation_history[-settings.LLM_HISTORY_TURNS :])
    messages.append({"role": "user", "content": user_message})
    return messages


# ---------------------------------------------------------------------------
# Streaming entry points (used by server.py)
# ---------------------------------------------------------------------------


async def prepare_voice_stream(
    call_sid: str, user_text: str
) -> tuple[list[dict], list[dict], str, bool]:
    """
    Run the per-turn pre-LLM pipeline:
      transcript → (rewrite + RAG retrieval) → assembled messages.

    Returns (messages, rag_docs, rewritten_query, rag_skipped).
    Records the user turn in conversation_store; the assistant turn is
    written by the caller after streaming finishes.
    """
    t_start = time.perf_counter()
    user_text = user_text.strip()[:500]

    history = conversation_store.get(call_sid)
    pl = _session_loggers.get(call_sid) or init_session_logger(call_sid)

    pl.log_transcript(user_text)

    rag_skipped = False
    rewritten_query = user_text

    if _should_skip_rag(user_text, history):
        logger.info(
            "[%s] Skipping RAG — conversational message: %s", call_sid, user_text
        )
        rag_docs: list[dict] = []
        rag_skipped = True
        pl.log_refined_query(user_text, "__SKIPPED__")
        pl.log_rag_results([])
    else:
        search_query = await _rewrite_query(call_sid, user_text, history)
        rewritten_query = search_query
        pl.log_refined_query(user_text, search_query)

        t_rag_start = time.perf_counter()
        rag_docs = await vector_store.retrieve(search_query, top_k=settings.VOICE_TOP_K)
        t_rag_ms = (time.perf_counter() - t_rag_start) * 1000
        logger.info(
            "[%s] LATENCY vector_search=%.0fms  docs=%d",
            call_sid,
            t_rag_ms,
            len(rag_docs),
        )
        pl.log_rag_results(rag_docs)

    messages = _build_voice_messages(user_text, rag_docs, history)
    pl.log_llm_context(messages)

    conversation_store.add(call_sid, "user", user_text)

    t_ms = (time.perf_counter() - t_start) * 1000
    logger.info("[%s] LATENCY prepare_voice_stream=%.0fms", call_sid, t_ms)
    return messages, rag_docs, rewritten_query, rag_skipped


_SENTENCE_END_RE = re.compile(r"([.!?]+(?:\s+|$))")


async def stream_voice_tokens(call_sid: str, messages: list[dict]):
    """
    Yield TTS-safe text chunks from the LLM.

    Strategy — "fast first, smooth after":
      - First chunk: flush on the first comma/period/colon/semicolon, OR after
        ~8 words, OR at 60 chars. Whichever comes first. Minimizes time-to-
        first-audio so the caller hears something quickly.
      - Subsequent chunks: flush on sentence boundaries (. ! ?) for natural
        TTS pacing.
    """
    client = get_session_client(call_sid)
    try:
        stream = await client.chat.completions.create(
            model=_VOICE_MODEL,
            messages=messages,
            stream=True,
            max_tokens=settings.VOICE_MAX_TOKENS,
            temperature=settings.VOICE_TEMPERATURE,
            extra_body={"keep_alive": settings.OLLAMA_KEEP_ALIVE},
        )
    except Exception as exc:
        logger.warning(
            "[%s] Pinned Ollama failed, swapping to fallback: %s", call_sid, exc
        )
        client = swap_session_client(call_sid)
        stream = await client.chat.completions.create(
            model=_VOICE_MODEL,
            messages=messages,
            stream=True,
            max_tokens=settings.VOICE_MAX_TOKENS,
            temperature=settings.VOICE_TEMPERATURE,
            extra_body={"keep_alive": settings.OLLAMA_KEEP_ALIVE},
        )

    buffer = ""
    first_chunk_sent = False
    word_count = 0

    FIRST_FLUSH_WORDS = settings.TTS_FIRST_FLUSH_WORDS
    FIRST_FLUSH_CHARS = settings.TTS_FIRST_FLUSH_CHARS

    async for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if not token:
            continue

        cleaned_token = _clean_token_for_tts(token)
        if not cleaned_token:
            continue

        buffer += cleaned_token
        word_count += cleaned_token.count(" ")

        if not first_chunk_sent:
            early_break = re.search(r"[,.:;!?]\s", buffer)
            word_flush = word_count >= FIRST_FLUSH_WORDS
            char_flush = len(buffer) >= FIRST_FLUSH_CHARS

            if early_break:
                end_idx = early_break.end()
                chunk_text = buffer[:end_idx].strip()
                buffer = buffer[end_idx:]
                word_count = buffer.count(" ")
            elif word_flush or char_flush:
                chunk_text = buffer.strip()
                buffer = ""
                word_count = 0
            else:
                continue

            if chunk_text and any(c.isalnum() for c in chunk_text):
                yield f"{chunk_text} "
                first_chunk_sent = True
        else:
            while True:
                match = _SENTENCE_END_RE.search(buffer)
                if not match:
                    break
                end_idx = match.end()
                sentence = buffer[:end_idx].strip()
                buffer = buffer[end_idx:]
                if sentence and any(c.isalnum() for c in sentence):
                    yield f"{sentence} "

    remainder = buffer.strip()
    if remainder:
        if remainder[-1] not in ".!?":
            remainder += "."
        yield f"{remainder} "
