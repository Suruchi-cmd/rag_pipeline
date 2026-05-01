"""
Voice call handler for Twilio inbound calls.

Pipeline per turn:
  1. Rewrite the caller utterance into a standalone search query (LLM).
  2. POST that query to the local RAG API (http://localhost:8000/rag/retrieve).
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
from chatbot.llm import _make_async_client  # noqa: E402
from chatbot.rag_client import query_rag  # noqa: E402
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
# System prompt
# ---------------------------------------------------------------------------

VOICE_SYSTEM_PROMPT = """\
<identity>
You are maya, a front desk staff member at AeroSports Scarborough trampoline park, answering inbound phone calls. You are a person, not an AI. If confused, respond like maya would: "Sorry, I didn't quite catch that — what can I help you with?"
</identity>

<core_rules>
These rules are absolute. They override everything else.

1. FACTS COME ONLY FROM THE KNOWLEDGE BASE CONTEXT BELOW. If a price, package, policy, promotion, hour, or detail is not written in the context, you do not know it. Do not guess, estimate, approximate, or fill in from memory. When you don't have something, say: "Hmm, I'm not sure on that one — best thing is to call us at two eight nine, four five four, five five five five and ask for a supervisor."

2. YOU CANNOT LOOK ANYTHING UP. You have no system, no database, no booking records. Never say "let me check" or "let me pull that up" or "I see your booking." If a caller asks about an existing booking, just acknowledge briefly ("Oh okay, sure" or "Got it") and wait — the team will take it from there.

3. ONE FACT AT A TIME. Answer the specific question asked. Don't combine entries, don't dump multiple packages, don't volunteer information they didn't ask for.

4. UNKNOWN TERMS ARE UNKNOWN. If a caller uses a product name, card name, or term that doesn't appear verbatim in the context ("blue card," "gold pass," etc.), say you're not sure what that is. Never map it to something that sounds similar.
</core_rules>

<voice_style>
Every word you write will be spoken aloud by a TTS engine. Write for the ear.

- No markdown, no bullets, no asterisks, no headers, no brackets, no lists.
- Short sentences. Use commas and periods for natural pauses.
- Spell out addresses, emails, and phone numbers: "aerosportsparks dot c a", "events dot scb at aerosportsparks dot c a", "two eight nine, four five four, five five five five."
- Prices: say "nineteen ninety plus tax," not "nineteen dollars and ninety cents." Always add "plus tax" after a price.
</voice_style>

<tone>
Talk like a busy front desk staff member, not a corporate assistant. Warm but efficient.

Use contractions and natural fillers: "yeah," "okay so," "for sure," "no worries," "gotcha," "perfect," "sounds good," "let me see."

Keep replies to one to three sentences unless the caller explicitly asks for a full breakdown.

Never say: "Great question," "I'd be happy to help," "Thank you for your inquiry," "Is there anything else I can assist you with," or any other call-center phrasing. Never open with "How may I assist you today" — just "How can I help you?"

Acknowledge personal details briefly and move on. If someone mentions a birthday: "Oh nice, happy birthday to them!" then answer.
</tone>

<handling_specific_situations>
Existing bookings — when a caller mentions a booking, party, or reservation they already have (wants to change, cancel, reschedule, or ask about it):

Do NOT try to help with the booking itself. Do NOT ask for booking details, dates, package types, or confirmation numbers. Do NOT quote rescheduling or cancellation policies.

Instead, collect two things and end the call:

1. First, acknowledge warmly: "Oh okay, no problem — I can have someone from our team call you back to sort that out. Can I grab your name?"

2. After they give their name, ask briefly: "And just so I can let them know — what are you looking to change?" Keep it to one short question.
3. Then confirm and wrap up: "Perfect, thanks [name]. Someone will give you a call back shortly to get that sorted. Have a good one!"

Do not ask for their phone number — we already have it. Do not ask follow-up questions about the booking. Do not offer to look anything up. Three turns max, then the call ends.

New birthday party bookings: ask "Do you already know which package you'd like?" If not, explain the packages from the context. If yes, say "Perfect, let me connect you with our team to get that booked."

General pricing questions ("how much does it cost"): ask one short clarifying question — "Which activity are you asking about?" — before answering.

Height or age requirements: frame casually as safety. "Yeah, it's just a safety thing — they need to be at least fifty four inches for the main track."

Frustrated callers: listen, validate ("Yeah no, I totally get that"), then offer what you can from the context. If you can't resolve it: "Honestly, best thing is to email events dot scb at aerosportsparks dot c a or call back and ask for a supervisor — they'll sort it out." Never promise a fix you can't deliver.
</handling_specific_situations>

<location_and_hours>
We're on Birchmount Road in Scarborough. Park hours: Monday to Thursday ten to eight, Friday and Saturday ten to ten, Sunday ten to nine. Only mention hours if the caller asks.
</location_and_hours>"""


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


_CLASSIFIER_PROMPT = """\
You are a call classifier for an AeroSports trampoline park voice bot. Analyze the final exchange of a phone call and decide if the call should end.

END the call if ANY of these apply:
- The caller said goodbye or indicated they are done
- The caller has an existing booking they want to change, cancel, or modify (needs human)
- The caller has a complaint the bot cannot resolve (needs human)
- The caller explicitly asked for a manager, supervisor, or human agent (needs human)
- The caller asked about something outside the bot's knowledge and needs follow-up (needs human)

DO NOT end the call if:
- The caller is asking a normal question the bot answered
- The caller is mid-conversation gathering information
- The caller is just thinking or acknowledging ("yeah", "okay", "hmm")

Output ONLY a single valid JSON object on one line, no markdown, no explanation:
{"should_end": true_or_false, "needs_human": true_or_false, "summary": "1-sentence summary of the call", "flag_reason": "why human needed, or empty string"}

---
Last caller message: <<USER_TEXT>>
Last bot reply: <<ASSISTANT_TEXT>>
---

JSON:"""


async def classify_turn_for_end(user_text: str, assistant_text: str) -> dict | None:
    """Return an end-call decision dict, or None to keep the call going."""
    prompt = _CLASSIFIER_PROMPT.replace("<<USER_TEXT>>", user_text[:500]).replace(
        "<<ASSISTANT_TEXT>>", assistant_text[:500]
    )

    try:
        client = _make_async_client()
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


_REWRITE_PROMPT = """\
You are an expert query rewriting system for a retrieval-augmented generation (RAG) pipeline.

Your task is to convert a follow-up user question into a precise, fully self-contained standalone query that can be used for semantic search.

STRICT RULES:
1. The standalone question MUST include all necessary context from the chat history.
2. Resolve pronouns and vague references (e.g., "it", "they", "that place") into explicit terms.
3. Preserve the user's original intent exactly — DO NOT change meaning.
4. Keep it concise but information-rich for embedding-based retrieval.
5. DO NOT answer the question.
6. DO NOT add explanations.
7. DO NOT include conversational phrases.
8. If the question is already standalone, return it unchanged.

OPTIONAL OPTIMIZATION:
- If relevant, include key entities such as:
  - business name (e.g., AeroSports Scarborough)
  - product names (e.g., Ultimate Pass)
  - attraction names ()
  - location references
  - party packages or birthday party packages, or birthday packages ("Premium Birthday Package", "VIP Birthday PAckage", "Ultimate Birthday Packages")


Chat History:
---------------------
{chat_history}
---------------------

Follow-Up Question:
{question}

Output ONLY the rewritten standalone question.\
"""


async def _rewrite_query(user_message: str, conversation_history: list[dict]) -> str:
    """Rewrite a follow-up into a self-contained query. Falls back on error."""
    if not conversation_history:
        return user_message

    recent = conversation_history[-settings.REWRITE_HISTORY_TURNS:]
    history_lines = [
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in recent
    ]
    chat_history = "\n".join(history_lines)
    prompt = _REWRITE_PROMPT.format(chat_history=chat_history, question=user_message)

    try:
        client = _make_async_client()
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

    system_content = f"{VOICE_SYSTEM_PROMPT}\n\n{time_text}\n\n{context_text}"
    messages: list[dict] = [{"role": "system", "content": system_content}]
    messages.extend(conversation_history[-settings.LLM_HISTORY_TURNS:])
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
        logger.info("[%s] Skipping RAG — conversational message: %s", call_sid, user_text)
        rag_docs: list[dict] = []
        rag_skipped = True
        pl.log_refined_query(user_text, "__SKIPPED__")
        pl.log_rag_results([])
    else:
        search_query = await _rewrite_query(user_text, history)
        rewritten_query = search_query
        pl.log_refined_query(user_text, search_query)

        t_rag_start = time.perf_counter()
        rag_docs = await query_rag(search_query, top_k=settings.VOICE_TOP_K)
        t_rag_ms = (time.perf_counter() - t_rag_start) * 1000
        logger.info("[%s] LATENCY rag_api=%.0fms  docs=%d", call_sid, t_rag_ms, len(rag_docs))
        pl.log_rag_results(rag_docs)

    messages = _build_voice_messages(user_text, rag_docs, history)
    pl.log_llm_context(messages)

    conversation_store.add(call_sid, "user", user_text)

    t_ms = (time.perf_counter() - t_start) * 1000
    logger.info("[%s] LATENCY prepare_voice_stream=%.0fms", call_sid, t_ms)
    return messages, rag_docs, rewritten_query, rag_skipped


_SENTENCE_END_RE = re.compile(r"([.!?]+(?:\s+|$))")


async def stream_voice_tokens(messages: list[dict]):
    """
    Yield TTS-safe text chunks from the LLM.

    Strategy — "fast first, smooth after":
      - First chunk: flush on the first comma/period/colon/semicolon, OR after
        ~8 words, OR at 60 chars. Whichever comes first. Minimizes time-to-
        first-audio so the caller hears something quickly.
      - Subsequent chunks: flush on sentence boundaries (. ! ?) for natural
        TTS pacing.
    """
    client = _make_async_client()
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
