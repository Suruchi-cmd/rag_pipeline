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

# RAG docs below this cosine similarity are dropped before building the prompt.
# /rag/retrieve has no server-side cutoff (unlike /rag/query), so low-relevance
# chunks would otherwise pollute the context. Keep at least one doc so the LLM
# still gets some context rather than a blank knowledge base section.
_MIN_RAG_SCORE: float = 0.5
_VOICE_STREAM_TIMEOUT: float = 25.0  # seconds; prevents dead-air if Ollama hangs


# ---------------------------------------------------------------------------
# ASR transcript normalization
# ---------------------------------------------------------------------------
# Common Deepgram mishearings of domain terms. Extend from logs as new variants
# appear — check logs/{call_sid}.log for STEP 1 transcripts that look wrong.
# Keys are case-insensitive; matching uses word boundaries so "aero" inside
# "aerodynamics" is not touched.
_ASR_ALIASES: dict[str, str] = {
    # Brand
    "arrow sports": "AeroSports",
    "ear sports": "AeroSports",
    "ero sports": "AeroSports",
    "aero sport": "AeroSports",
    "arrow sport": "AeroSports",
    # Location
    "birth mount": "Birchmount",
    "birch month": "Birchmount",
    "birch mount": "Birchmount",
    "scarboro": "Scarborough",
    # Attractions
    "donut slid": "Donut Slide",
    "carpet slid": "Carpet Slide",
    "ninja maize": "Ninja Maze",
    "ninja warrior course": "Ninja Warrior",
    "slam basketball": "Aero Slam",
    # Passes
    "v i p pass": "VIP Pass",
    "v i p party": "VIP Party",
    "vip jump": "VIP Jump Pass",
    "all-day pass": "All Day Pass",
    "all day past": "All Day Pass",
    "thirty day pass": "30 Day Pass",
    "ninety day pass": "90 Day Pass",
    # Programs
    "p a day camp": "PA Day Camp",
    "march brake camp": "March Break Camp",
    "march break cap": "March Break Camp",
    "toddler tile": "Toddler Time",
    # Birthday packages
    "premium birthday": "Premium Birthday Package",
    "vip birthday": "VIP Birthday Package",
    "ultimate birthday": "Ultimate Birthday Package",
}

# Pre-compile patterns once at module load. Sort longest-first so multi-word
# phrases match before any single-word alias they might contain.
_ASR_ALIAS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE), right)
    for wrong, right in sorted(
        _ASR_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True
    )
]


def _normalize_transcript(text: str) -> str:
    """Map known ASR mishearings to canonical domain terms.

    Two-stage protection: explicit alias map (this function) catches the
    common cases; the rewriter then resolves anything else against context.
    Extend _ASR_ALIASES from real transcripts in logs/{call_sid}.log.
    """
    if not text:
        return text
    normalized = text
    for pattern, right in _ASR_ALIAS_PATTERNS:
        normalized = pattern.sub(right, normalized)
    return normalized


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

_AGENT_NAME_LOWER = settings.AGENT_NAME.lower()

VOICE_SYSTEM_PROMPT = f"""\
<identity>
You are {_AGENT_NAME_LOWER}, a front desk staff member at {settings.BUSINESS_NAME} trampoline park, answering inbound phone calls. You are a person, not an AI. If confused, respond like {_AGENT_NAME_LOWER} would: "Sorry, I didn't quite catch that — what can I help you with?"
</identity>

<core_rules>
These rules are absolute. They override everything else.

1. Keep replies short and conversational — usually one to three sentences. Match the question: quick facts get quick answers, richer questions get a sentence of context plus a follow-up. Never wall-of-text or list everything you know.

2. FACTS COME ONLY FROM THE KNOWLEDGE BASE CONTEXT BELOW. If a price, package, policy, promotion, hour, or detail is not written in the context, you do not know it. Do not guess, estimate, approximate, or fill in from memory. When you don't have something, say: "Hmm, I'm not sure on that one — best thing is to call us at {settings.BUSINESS_PHONE_SPOKEN} and ask for a supervisor."

3. YOU CANNOT LOOK ANYTHING UP. You have no system, no database, no booking records. Never say "let me check" or "let me pull that up" or "I see your booking." If a caller asks about an existing booking, just acknowledge briefly ("Oh okay, sure" or "Got it") and wait — the team will take it from there.

4. ONE FACT AT A TIME. Answer the specific question asked. Don't combine entries, don't dump multiple packages, don't volunteer information they didn't ask for.

5. UNKNOWN TERMS ARE UNKNOWN. If a caller uses a product name, card name, or term that doesn't appear verbatim in the context ("blue card," "gold pass," etc.), say you're not sure what that is. Never map it to something that sounds similar.
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

Sound like a real person on the phone — warm, brief, conversational. Acknowledge the question lightly ("yeah", "for sure", "totally"), give the answer, and check if they want more when it makes sense. Short answer for a short question; a touch of context for a richer one. Never lecture.

Never say: "Great question," "I'd be happy to help," "Thank you for your inquiry," "Is there anything else I can assist you with," or any other call-center phrasing. Never open with "How may I assist you today" — just "How can I help you?"

Acknowledge personal details briefly and move on. If someone mentions a birthday: "Oh nice, happy birthday to them!" then answer.
</tone>

<response_pattern>
Conversation should flow naturally — like a friendly front-desk chat, not an interrogation.

Pattern: light acknowledgment → answer → optional follow-up question when the answer opens choices. For multi-option questions, name the options briefly and ask one narrowing question instead of dumping the full menu. For simple factual questions, just answer.

Caller: "What birthday packages do you have?"
You: "Yeah, we've got three — Premium, VIP, and Ultimate. How many kids are you thinking?"

Caller: "How much does it cost to come in?"
You: "Depends on what you're after — are you looking at a single session, or more of an all-day visit?"

Caller: "What time do you close Friday?"
You: "Friday we're open till ten."

Caller: "Do you guys have go-karts?"
You: "Yeah, we've got a main track and a mini one — were you looking for adults or kids?"

Caller: "Where are you located?"
You: "We're on Birchmount Road in Scarborough."

One to three sentences. Match the energy of the question — short asks get short answers; broader asks get a sentence of context plus a follow-up.
</response_pattern>

<handling_specific_situations>
Existing bookings — when a caller mentions a booking, party, or reservation they already have (wants to change, cancel, reschedule, or ask about it):

Do NOT try to help with the booking itself. Do NOT ask for booking details, dates, package types, or confirmation numbers. Do NOT quote rescheduling or cancellation policies.

Instead, collect two things and end the call:

1. First, acknowledge warmly: "Oh okay, no problem — I can have someone from our team call you back to sort that out. Can I grab your name?"

2. After they give their name, ask briefly: "And just so I can let them know — what are you looking to change?" Keep it to one short question.
3. Then confirm and wrap up: "Perfect, thanks [name]. Someone will give you a call back shortly to get that sorted. Have a good one!"

Do not ask for their phone number — we already have it. Do not ask follow-up questions about the booking. Do not offer to look anything up. Three turns max, then the call ends.

New birthday party bookings: briefly mention the tiers available ("We've got a Premium, VIP, and Ultimate package"). Ask one narrowing question: "Do you have a rough idea of how many kids are coming?" or "Do you already know which one you'd like?" Once they pick one, say "Perfect — for bookings you'd want to email {settings.BUSINESS_EMAIL_SPOKEN} or I can have our events team reach out."

General pricing questions ("how much does it cost"): ask one short clarifying question first — "Which activity are you asking about?" or "Are you thinking a single visit or something longer-term?" — before answering.

Height or age requirements: frame casually as safety. "Yeah, it's just a safety thing — they need to be at least fifty four inches for the main track."

Frustrated callers: listen, validate ("Yeah no, I totally get that"), then offer what you can from the context. If you can't resolve it: "Honestly, best thing is to email {settings.BUSINESS_EMAIL_SPOKEN} or call back and ask for a supervisor — they'll sort it out." Never promise a fix you can't deliver.
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


# Pre-compile word-boundary regex for each keyword list. Substring `in` matched
# inside other words ("i booked" matched "would if i booked it") and triggered
# false captures. \b anchors prevent that.
def _compile_keyword_re(keywords: list[str]) -> re.Pattern:
    return re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b",
        re.IGNORECASE,
    )


_END_CALL_DEFINITE_RE = _compile_keyword_re(_END_CALL_KEYWORDS_DEFINITE)
_END_CALL_MAYBE_RE = _compile_keyword_re(_END_CALL_KEYWORDS_MAYBE)
_BOOKING_CAPTURE_RE = _compile_keyword_re(_BOOKING_CAPTURE_TRIGGERS)


def check_end_keywords(user_text: str) -> str:
    """Return 'definite' | 'maybe' | 'none' for the user's last utterance."""
    text = user_text.strip()
    if not text:
        return "none"
    if _END_CALL_DEFINITE_RE.search(text):
        return "definite"
    if _END_CALL_MAYBE_RE.search(text):
        return "maybe"
    return "none"


def check_booking_capture_trigger(user_text: str) -> bool:
    text = user_text.strip()
    if not text:
        return False
    return bool(_BOOKING_CAPTURE_RE.search(text))


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

        # Strip Qwen3 reasoning blocks BEFORE JSON extraction — otherwise the
        # greedy `\{.*\}` regex captures `{...}</think>{...}` and json.loads dies.
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        raw = raw.replace("</think>", "").strip()

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


# Tokens that typically signal the message depends on prior context (pronoun,
# ellipsis, fragment-response). When ANY appear → run the LLM rewriter.
# Otherwise the message is probably self-contained and we can skip the hop.
# Conservative on purpose: when in doubt, rewrite.
_ANAPHORA_RE = re.compile(
    r"\b("
    r"it|its|they|them|their|theirs|"
    r"that|those|these|this|"
    r"one|ones|same|other|another|"
    r"the\s+(one|other|same|first|second|last)|"
    r"how\s+about|what\s+about|"
    r"and\s+(you|the|on|in|for|at|sunday|monday|tuesday|wednesday|thursday|friday|saturday|today|tomorrow)|"
    r"tell\s+me\s+more|"
    r"more\s+(info|details|about)"
    r")\b",
    re.IGNORECASE,
)


def _likely_needs_rewrite(user_message: str) -> bool:
    """Heuristic: only rewrite when message shows context dependence.

    True if the message contains a pronoun / elliptical phrase OR is very
    short (sentence fragments are usually responses to clarifying questions).
    Saves a ~300-1000ms LLM hop on roughly 30-50% of turns where the user
    asks a fully-formed question.
    """
    if len(user_message.split()) < 3:
        return True  # short fragments almost always depend on context
    return bool(_ANAPHORA_RE.search(user_message))


_REWRITE_PROMPT = """\
Rewrite a follow-up question into a standalone search query for a knowledge base.

Rules:
1. Resolve pronouns and ellipsis using history ("it", "that one", "and Sunday").
2. If the question already stands alone, return it UNCHANGED.
3. DO NOT add prices, facts, or details from history into the query.
4. DO NOT change the user's intent or topic.
5. DO NOT answer the question.
6. Output ONE line, plain text, no quotes, no labels.

Examples:

History:
User: What birthday packages do you have?
Assistant: Premium, VIP, and Ultimate.
Question: How much is the VIP one?
Standalone: How much does the VIP birthday package cost?

History:
User: What time do you open Friday?
Assistant: Ten in the morning.
Question: And Sunday?
Standalone: What time do you open Sunday?

History:
User: Tell me about the go-karts.
Assistant: We have a main and a mini track.
Question: How fast do they go?
Standalone: How fast do the go-karts go?

History:
User: I want to book a birthday party.
Assistant: Sure, how many kids?
Question: Around fifteen.
Standalone: Birthday party for fifteen kids.

History:
User: How much is the all-day pass?
Assistant: Thirty nine ninety plus tax.
Question: Where are you located?
Standalone: Where are you located?

History:
{chat_history}
Question: {question}
Standalone:"""


# ── Output sanitization ─────────────────────────────────────────────────────

# Strip Qwen3-style reasoning blocks (closed, unclosed, orphan close tags).
_REWRITE_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_REWRITE_PARTIAL_THINK_RE = re.compile(r"<think>.*", re.DOTALL)

# Models often prefix the answer with a label like "Standalone:" or wrap in quotes.
_REWRITE_LABEL_RE = re.compile(
    r"^\s*(standalone(?:\s+question)?|rewritten(?:\s+question)?|query|search\s+query)\s*[:\-]\s*",
    re.IGNORECASE,
)
_REWRITE_QUOTE_WRAP_RE = re.compile(r'^["\'`]+\s*(.+?)\s*["\'`]+$')

# Substrings that indicate prompt leakage, refusal, or chain-of-thought spillover.
_REWRITE_GARBAGE_TOKENS: tuple[str, ...] = (
    "chat history:",
    "follow-up question",
    "strict rules",
    "examples:",
    "i cannot",
    "i can't",
    "i'm unable",
    "i don't have",
    "as an ai",
    "<think",
    "</think",
)


def _sanitize_rewrite(raw: str) -> str:
    """Strip think-blocks, labels, quotes, markdown; take first non-empty line."""
    text = _REWRITE_THINK_BLOCK_RE.sub("", raw)
    text = _REWRITE_PARTIAL_THINK_RE.sub("", text)
    text = text.replace("</think>", "")
    text = text.strip()

    # Multi-line outputs usually mean the model added an explanation after the
    # rewrite. Take the first non-empty, non-label line.
    for candidate in text.splitlines():
        candidate = candidate.strip()
        if not candidate:
            continue
        candidate = _REWRITE_LABEL_RE.sub("", candidate).strip()
        if candidate:
            text = candidate
            break

    # Strip surrounding quotes/backticks the model may add.
    m = _REWRITE_QUOTE_WRAP_RE.match(text)
    if m:
        text = m.group(1)

    # Strip markdown bold/italic markers.
    text = text.replace("**", "").replace("__", "").strip("*_`")

    return text.strip()


def _rewrite_is_garbage(rewritten: str) -> bool:
    """Reject empty, oversized, or prompt-leak / refusal outputs."""
    if not rewritten:
        return True
    if len(rewritten) < 3 or len(rewritten) > 300:
        return True
    lowered = rewritten.lower()
    return any(tok in lowered for tok in _REWRITE_GARBAGE_TOKENS)


async def _rewrite_query(user_message: str, conversation_history: list[dict]) -> str:
    """Rewrite a follow-up into a self-contained query. Falls back on any failure.

    Bulletproof: every failure path returns the original message intact, so
    a broken rewriter never starves RAG of a query.
    """
    # Guard 1: empty / whitespace-only — nothing to rewrite, nothing to search.
    if not user_message or not user_message.strip():
        return user_message

    # Guard 2: no conversation context — nothing to resolve against.
    if not conversation_history:
        return user_message

    # Guard 3: heuristic skip — message has no pronouns/ellipsis, almost
    # certainly self-contained. Saves the LLM hop. The rewriter would have
    # returned UNCHANGED anyway in these cases per Rule 2 of the prompt.
    if not _likely_needs_rewrite(user_message):
        logger.info(
            "Rewrite skipped — no anaphora detected in: %r", user_message[:100]
        )
        return user_message

    # Build history block. Truncate per-message to keep prompt size bounded
    # under load — long assistant replies can blow past the model's effective
    # context window when chained with the system prompt + RAG context later.
    recent = conversation_history[-settings.REWRITE_HISTORY_TURNS :]
    history_lines = [
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
        for m in recent
    ]
    chat_history = "\n".join(history_lines) if history_lines else "(no prior turns)"
    prompt = _REWRITE_PROMPT.format(chat_history=chat_history, question=user_message)

    try:
        client = _make_async_client()
        response = await client.chat.completions.create(
            model=_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=settings.REWRITE_MAX_TOKENS,
            temperature=settings.REWRITE_TEMPERATURE,
            # Stop sequences keep the model from continuing past the rewrite
            # into another fake example block. Labels MUST match the actual
            # prompt section headers — _REWRITE_PROMPT uses "History:" not
            # "Chat history:", so stop on "History:" / "Question:".
            stop=["\nHistory:", "\n\nHistory:", "\nQuestion:", "\n\nQuestion:"],
        )
        raw = response.choices[0].message.content or ""
        rewritten = _sanitize_rewrite(raw)

        if _rewrite_is_garbage(rewritten):
            logger.warning(
                "Rewrite rejected (garbage/leak); using original. raw=%r cleaned=%r",
                raw[:200],
                rewritten[:200],
            )
            return user_message

        if rewritten == user_message:
            logger.info("Rewrite returned unchanged: %r", user_message[:100])
        else:
            logger.info("Rewritten query: %r → %r", user_message[:100], rewritten[:200])
        return rewritten
    except Exception as exc:
        logger.error("Query rewrite failed, using original: %s", exc)
        return user_message


# ---------------------------------------------------------------------------
# TTS cleaning
# ---------------------------------------------------------------------------
# Patterns pre-compiled once at module load. _clean_token_for_tts runs on
# every streamed token (potentially hundreds per turn), so re.compile per
# call would burn measurable CPU under load.

_TOK_STAR_RE = re.compile(r"\*+")
_TOK_HASH_RE = re.compile(r"#+")
_TOK_BACKTICK_RE = re.compile(r"`+")

_TTS_THINK_FULL_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TTS_THINK_PARTIAL_RE = re.compile(r"<think>.*", re.DOTALL)
_TTS_HASH_LEAD_RE = re.compile(r"#+\s*")
_TTS_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_TTS_LIST_DASH_RE = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
_TTS_HIGH_EMOJI_RE = re.compile(r"[\U00010000-\U0010FFFF]")
_TTS_LOW_EMOJI_RE = re.compile(r"[☀-➿︀-️]")
_TTS_MINS_RE = re.compile(r"\bmins?\b", re.IGNORECASE)
_TTS_HRS_RE = re.compile(r"\bhrs?\b", re.IGNORECASE)
_TTS_PRICE_DECIMAL_RE = re.compile(r"\$(\d+)\.(\d{2})")
_TTS_PRICE_WHOLE_RE = re.compile(r"\$(\d+)")
_TTS_BARE_DECIMAL_RE = re.compile(r"\b(\d+)\.(\d{2})\b")
_TTS_NEWLINE_RE = re.compile(r"\n+")


def _clean_token_for_tts(token: str) -> str:
    """Per-token cleaner — strip markdown symbols only, keep whitespace.

    NOTE: We do NOT strip `$` here because most modern TTS engines
    (ElevenLabs included) read "$39" as "thirty-nine dollars" naturally.
    Stripping would leave bare "39" which the TTS reads without the unit.
    Full price normalization happens later in clean_for_tts when the
    accumulated text is written to history.
    """
    token = _TOK_STAR_RE.sub("", token)
    token = _TOK_HASH_RE.sub("", token)
    token = _TOK_BACKTICK_RE.sub("", token)
    return token


def clean_for_tts(text: str) -> str:
    """Full cleaner for accumulated text written to conversation history."""
    # Strip Qwen3-style <think>…</think> reasoning blocks (closed, unclosed, orphaned).
    text = _TTS_THINK_FULL_RE.sub("", text)
    text = _TTS_THINK_PARTIAL_RE.sub("", text)
    text = text.replace("</think>", "")
    text = _TOK_STAR_RE.sub("", text)
    text = _TTS_HASH_LEAD_RE.sub("", text)
    text = _TOK_BACKTICK_RE.sub("", text)
    text = _TTS_MD_LINK_RE.sub(r"\1", text)
    text = _TTS_LIST_DASH_RE.sub("", text)
    # TTS reads emojis as their text description ("party popper") — strip them.
    text = _TTS_HIGH_EMOJI_RE.sub("", text)
    text = _TTS_LOW_EMOJI_RE.sub("", text)
    text = _TTS_MINS_RE.sub("minutes", text)
    text = _TTS_HRS_RE.sub("hours", text)
    # "$39.90" / "$39" / bare "39.90" → spoken form "39 90" / "39 dollars".
    text = _TTS_PRICE_DECIMAL_RE.sub(r"\1 \2", text)
    text = _TTS_PRICE_WHOLE_RE.sub(r"\1 dollars", text)
    text = _TTS_BARE_DECIMAL_RE.sub(r"\1 \2", text)
    text = _TTS_NEWLINE_RE.sub(" ", text).strip()
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
    messages.extend(conversation_history[-settings.LLM_HISTORY_TURNS :])
    messages.append({"role": "user", "content": user_message})
    return messages


# ---------------------------------------------------------------------------
# Streaming entry points (used by server.py)
# ---------------------------------------------------------------------------


async def prepare_voice_stream(call_sid: str, user_text: str) -> list[dict]:
    """
    Run the per-turn pre-LLM pipeline:
      transcript → (rewrite + RAG retrieval) → assembled messages.

    Records the user turn in conversation_store; the assistant turn is
    written by the caller after streaming finishes (it depends on how much
    was actually spoken before any interruption).
    """
    t_start = time.perf_counter()
    raw_text = user_text.strip()[:500]
    user_text = _normalize_transcript(raw_text)
    if user_text != raw_text:
        logger.info(
            "[%s] Transcript normalized: %r → %r",
            call_sid,
            raw_text[:200],
            user_text[:200],
        )

    history = await conversation_store.get(call_sid)
    pl = _session_loggers.get(call_sid) or init_session_logger(call_sid)

    pl.log_transcript(user_text)

    if _should_skip_rag(user_text, history):
        logger.info(
            "[%s] Skipping RAG — conversational message: %s", call_sid, user_text
        )
        rag_docs: list[dict] = []
        pl.log_refined_query(user_text, "__SKIPPED__")
        pl.log_rag_results([])
    else:
        search_query = await _rewrite_query(user_text, history)
        pl.log_refined_query(user_text, search_query)

        t_rag_start = time.perf_counter()
        rag_docs = await query_rag(search_query, top_k=settings.VOICE_TOP_K)
        t_rag_ms = (time.perf_counter() - t_rag_start) * 1000
        logger.info(
            "[%s] LATENCY rag_api=%.0fms  docs=%d", call_sid, t_rag_ms, len(rag_docs)
        )
        pl.log_rag_results(rag_docs)

        # Drop docs below the score threshold; /rag/retrieve returns all top_k
        # regardless of relevance. Keep at least one so context is never empty.
        if rag_docs:
            original_count = len(rag_docs)
            filtered = [d for d in rag_docs if d.get("score", 1.0) >= _MIN_RAG_SCORE]
            if filtered:
                rag_docs = filtered
                logger.info(
                    "[%s] RAG: %d/%d docs passed score threshold %.2f",
                    call_sid,
                    len(filtered),
                    original_count,
                    _MIN_RAG_SCORE,
                )
            else:
                rag_docs = rag_docs[:1]
                logger.warning(
                    "[%s] RAG: all %d docs below threshold %.2f — keeping top-1",
                    call_sid,
                    original_count,
                    _MIN_RAG_SCORE,
                )

    messages = _build_voice_messages(user_text, rag_docs, history)
    pl.log_llm_context(messages)

    await conversation_store.add(call_sid, "user", user_text)

    t_ms = (time.perf_counter() - t_start) * 1000
    logger.info("[%s] LATENCY prepare_voice_stream=%.0fms", call_sid, t_ms)
    return messages


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
        top_p=settings.LLM_TOP_P,
        extra_body={"keep_alive": settings.OLLAMA_KEEP_ALIVE},
        timeout=_VOICE_STREAM_TIMEOUT,
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
