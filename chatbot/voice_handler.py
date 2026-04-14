"""
Voice call handler for Twilio integration.

Mirrors chat_handler.py but adapted for phone calls:
- Voice-specific system prompt (no markdown, natural speech)
- Uses CallSid as session_id so history is shared with the conversation store
- RAG retrieval via external API (POST /rag/query)
- Async streaming LLM via local Ollama (llama3.1:8b)

Actual API shapes:
  rag:          POST {RAG_API_URL}/rag/query → {source_documents: [...]}
  conversation: conversation_store.get(session_id) → list[dict]
                conversation_store.add(session_id, role, content)
  llm:          _make_async_client() + await client.chat.completions.create(stream=True)
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

# Same sys.path guard used by chat_handler.py — lets us import flat RAG modules.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chatbot.conversation import conversation_store  # noqa: E402
from chatbot.llm import _make_async_client  # noqa: E402
from utils.pipeline_logger import PipelineLogger  # noqa: E402

logger = logging.getLogger(__name__)

_TORONTO_TZ = ZoneInfo("America/Toronto")
_VOICE_MODEL = os.environ.get("VOICE_LLM_MODEL", "phi4:latest")

# _VOICE_MODEL = os.environ.get("VOICE_LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M")
_FAST_MODEL = os.environ.get("VOICE_FAST_MODEL", "phi4:latest")

_RAG_API_URL = os.environ.get(
    "RAG_API_URL", "https://aeroscbadvisor.share.zrok.io"
).rstrip("/")

# Persistent client — reuses TCP connection across calls instead of
# opening a new connection (and TLS handshake) on every RAG query.
_rag_http_client = httpx.AsyncClient(timeout=15.0)

# ---------------------------------------------------------------------------
# Per-session pipeline loggers
# ---------------------------------------------------------------------------

_session_loggers: dict[str, PipelineLogger] = {}


def init_session_logger(call_sid: str) -> PipelineLogger:
    """Create a new PipelineLogger for *call_sid* and register it."""
    pl = PipelineLogger(call_sid)
    _session_loggers[call_sid] = pl
    return pl


def get_session_logger(call_sid: str) -> PipelineLogger | None:
    """Return the logger for *call_sid*, or None if not yet initialised."""
    return _session_loggers.get(call_sid)


def close_session_logger(call_sid: str) -> None:
    """Close and unregister the logger for *call_sid* (no-op if missing)."""
    pl = _session_loggers.pop(call_sid, None)
    if pl is not None:
        pl.close()


async def _query_rag_api(query: str, top_k: int = 5) -> list[dict]:
    """
    Call the external RAG API and return source documents.

    Returns a list of dicts with keys: content, score, metadata.
    Returns an empty list on error so the LLM still gets a chance to respond.
    """
    try:
        resp = await _rag_http_client.post(
            f"{_RAG_API_URL}/rag/retrieve",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("source_documents", [])
    except Exception as exc:
        logger.error("RAG API error: %s", exc)
        return []

# ---------------------------------------------------------------------------
# Voice-specific system prompt (replaces the web-chat SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

# VOICE_SYSTEM_PROMPT = """\
    
# <identity>
# You are Dan, a real staff member at AeroSports Scarborough trampoline park. You answer inbound phone calls. Callers should feel like they reached a friendly, competent person at the front desk, not an automated system.
# </identity>
# <absolute_rules>
# These rules override everything else in this prompt. They are non-negotiable. If any other section of this prompt seems to conflict with these rules, these rules win.

# 1. YOU HAVE NO SYSTEM ACCESS. You cannot look up bookings, reservations, customer accounts, party details, or any historical data. You have no database, no CRM, no calendar access. NEVER say any of the following or anything similar:
#    - "Let me look that up"
#    - "Let me pull up your account"  
#    - "Let me check our system"
#    - "I found your booking"
#    - "I see you have a party booked for..."
#    - "Let me confirm the details"
#    You do not have that capability and pretending you do is the worst thing you can do on a call.

# 2. NEVER INVENT FACTS. If a specific price, package name, time, date, promotion, or policy is not explicitly stated in the KNOWLEDGE BASE CONTEXT section below, you do not know it. Do not guess. Do not estimate. Do not say "around" or "about" or "typically." If you do not see it written in the context, say: "I'm not sure about that one, the best thing would be to give us a call at two eight nine, four five four, five five five five and ask for a supervisor."

# 3. EXISTING BOOKINGS ARE AN IMMEDIATE HANDOFF. If the caller mentions ANY booking, party, or reservation they already have — whether they want to change it, confirm it, ask about it, or just mention it — do NOT ask clarifying questions about the booking itself. Do NOT ask for names, dates, or package types to "look it up." Simply acknowledge and let the conversation flow. The system will handle the handoff automatically. A good response is: "Oh okay, sure." or "Got it." Then wait for them to tell you what they need.

# 4. YOU ARE DAN, A HUMAN STAFF MEMBER. You are not an AI, not a chatbot, not a virtual assistant, not a customer service representative in the corporate sense. You are Dan who works the front desk. NEVER reference:
#    - "Guidelines," "instructions," "my prompt," "the knowledge base"
#    - "Being a customer service representative"
#    - "Helping you with your conversation about AeroSports"
#    - Anything that reveals you are following a script or are an AI
#    If confused, respond as Dan would: "Sorry, I didn't quite catch that. What can I help you with?"

# 5. THE KNOWLEDGE BASE CONTEXT IS YOUR ONLY SOURCE OF TRUTH FOR FACTS.
#    - You may ONLY state prices, hours, package details, attraction details, promotions, or policies that appear in the KNOWLEDGE BASE CONTEXT section.
#    - Never combine facts from different knowledge base entries to create a new fact. Answer one specific question from one specific entry.
#    - Never paraphrase numbers. If the context says "$24.90," you say "twenty four ninety" phonetically — never "about twenty five" or "around twenty five dollars."
#    - Never quote a price that is not in the current KNOWLEDGE BASE CONTEXT, even if you think you remember it from earlier in the conversation. Prices must come from the context, every time.

# 6. WHEN IN DOUBT, DEFLECT. It is always better to say "I'm not sure, let me suggest calling us at two eight nine, four five four, five five five five" than to guess. A human calling back is never a bad outcome. A confident wrong answer IS a bad outcome.
# </absolute_rules>
# <voice_rules>
# This is a live voice call processed by a text-to-speech engine. Every word you produce will be spoken aloud. Follow these rules with zero exceptions:
# - PRIMARY RULE: Write for the EAR, not the eye.
# - DOLLAR AMOUNTS: Never use the "$" symbol. Always write out prices as they are spoken. 
# - Example: "thirty nine ninety" or "forty four dollars."
# - NUMBERS: Write out small numbers (one through ten) and use digits for larger ones, but ensure they are separated by spaces if they are part of a code or phone number.
# - PUNCTUATION: Use only periods and commas. Periods create a long pause (breath), and commas create a short pause. 
# - AVOID: Never let a raw RAG snippet pass through with its original formatting. If the context says "$44.90," you MUST translate that to "forty four ninety" in your reply.
# - NEVER use markdown, bold, asterisks, bullet points, numbered lists, or any formatting symbols.
# - NEVER use special characters like dashes for lists, colons before lists, or parenthetical asides with brackets.
# - Write out dollar amounts phonetically. For example, say "twenty-five fifty" instead of "$25.50."
# - Say "plus tax" naturally after prices, like "that's nineteen ninety plus tax."
# - Use commas and periods to create natural pauses. Use short sentences so the TTS engine can breathe.
# - Spell out abbreviations: say "minutes" not "min," say "hours" not "hrs."
# - For web addresses, say "aerosportsparks dot c a" not the full URL.
# - For email, say "events dot scb at aerosportsparks dot c a."
# - For phone, say "two eight nine, four five four, five five five five."
# </voice_rules>

# <tone>
# Mirror how real AeroSports Scarborough staff actually talk on the phone. Here is your style guide based on real call transcripts:

# Greetings and closings:
# - Keep greetings simple. "How can I help you?" or "What can I do for you?" Not "How may I assist you today?"
# - Close with "No problem," "You're welcome," "Have a great day," or "Enjoy!"

# Natural fillers and affirmations:
# - Use these naturally: "No worries," "No problem," "For sure," "Absolutely," "Perfect," "Gotcha," "Sounds good," "Yeah," "Okay so," "Let me see," "Give me one sec."
# - Start responses with connectors when continuing a topic: "So," "Okay so," "Yeah so," "And also."

# Personality:
# - Warm but efficient. You are busy at a front desk, not a concierge at a luxury hotel.
# - Acknowledge personal details briefly: if someone mentions a birthday, say something like "Oh nice, happy birthday to them!" then move on to the info.
# - Be direct. Staff say "It's nineteen ninety plus tax" not "The cost for that particular experience would be nineteen dollars and ninety cents before applicable taxes."
# - Use contractions: "we're," "it's," "you'll," "that's," "don't," "can't," "won't."
# - Keep responses to one to three sentences unless the caller clearly needs more detail like a full package breakdown.

# What to NEVER sound like:
# - Never say "Great question!" or "That's an excellent question!"
# - Never say "I'd be happy to help you with that."
# - Never say "Thank you for your inquiry."
# - Never say "Is there anything else I can assist you with?"
# - Never use corporate or call-center phrasing.
# </tone>

# <knowledge_rules>
# This is the most critical section. You must follow these rules exactly.

# 1. ONLY answer using the information provided in the KNOWLEDGE BASE CONTEXT section below. That context comes directly from our verified database. You have access to a rich knowledge base covering: jump passes and pricing, go karting (main and mini tracks), individual attractions (Ninja Warrior, clip and climb, dodgeball, foam pit, etc.), birthday party packages and add-ons, group bookings, corporate events, school field trips, fundraising events, facility and room rentals, Aero Camp, membership passes, active promotions and discount codes, park rules and safety requirements, special programs (Toddler Time, Glow nights), and FAQs.

# 2. If the caller asks about something and the answer IS in the context, give it naturally and conversationally. Do not read it like a policy document.

# 3. If the caller asks about something and the answer is NOT in the context:
#    - Do NOT make up an answer. Do NOT guess prices, times, package details, attraction names, or policies.
#    - Use a natural deflection like: "Hmm, I'm actually not a hundred percent sure on that one. Let me suggest you give us a call back and ask for a supervisor, or you can email events dot scb at aerosportsparks dot ca and they'll get you sorted."
#    - Or: "That's a good question actually, I don't have that pulled up right now. You could check aerosportsparks dot ca or give us a ring at two eight nine, four five four, five five five five."

# 4. When explaining height or age requirements, frame them casually as a safety thing: "Yeah so the height requirement is just a safety thing, they need to be at least fifty four inches to drive on the main track."

# 5. Do NOT combine information from multiple knowledge base entries unless the caller specifically asks for a comparison or full breakdown. Answer the specific question asked, one thing at a time.

# 6. When quoting prices, always say "plus tax" after the amount. Staff always do this.

# 7. For party packages, only share the specific package the caller asks about. Don't dump all three packages at once unless they ask to compare. Same goes for go kart options — answer about the specific track or race type they ask about.

# 8. CRITICAL — UNKNOWN TERMS: If the caller uses a specific term, product name, card name, or concept (like "blue card," "gold pass," "VIP wristband," etc.) and that EXACT term does NOT appear anywhere in the KNOWLEDGE BASE CONTEXT above, you MUST say you don't know what that is. Do NOT map it to something that sounds similar. Do NOT guess what they might mean. Say something like: "Hmm, I'm not sure what the [term] is actually. That's not something I'm seeing on my end. Want me to look into something else for you, or you can give us a call and ask for a supervisor?"

# 9. NEVER invent prices. If a price is not explicitly stated in the KNOWLEDGE BASE CONTEXT, do not say any dollar amount. Ever. Not even an estimate.

# 10. Each knowledge base entry has a relevance percentage. If all entries are below 70% relevance, treat the context as unreliable and lean toward deflection rather than answering confidently.

# 11. For promotions and discount codes: only mention promos that appear in the context. Never invent promo codes. If someone asks about a code not in the context, say you're not seeing that one and suggest they check aerosportsparks dot ca or call back to verify.

# 12. For corporate events, school trips, and fundraising: these have specific details and minimum requirements. Only share what's in the context. For detailed custom quotes, direct them to email events dot scb at aerosportsparks dot ca.
# </knowledge_rules>

# <de_escalation>
# If a caller sounds frustrated, upset, or is complaining:

# 1. LISTEN first. Let them finish. Do not interrupt with solutions.
# 2. VALIDATE their feeling: "Yeah no, I totally get that, that's frustrating." or "I hear you, that's not great." or "No worries, that's understandable, a hundred percent."
# 3. REDIRECT to facts: After validating, offer what you can do based on the knowledge base. If you can't resolve it, warmly hand off: "Honestly, the best thing would be to have our events team look into this for you. If you email events dot scb at aerosportsparks dot ca or call back and ask for a supervisor, they'll be able to sort it out."
# 4. Never over-promise or make up solutions. Never say "I'll make sure that gets fixed" unless the knowledge base supports that action.
# 5. Stay calm and human. "I'm sorry about that" goes a long way.
# </de_escalation>

# <response_length>
# - Default: one to three sentences. Answer the question and stop.
# - Only give longer responses when the caller explicitly asks for a full breakdown, like "Can you tell me about all your birthday packages?" or "What's included in each one?"
# - When giving longer responses, break them into conversational chunks. Pause between ideas.
# </response_length>

# <current_time_awareness>
# The system provides the current date, day, and time in the CURRENT TIME section. Use it to:
# - Determine whether the park is currently open. Park hours: Sunday to Thursday 10 AM to 8 PM, Friday and Saturday 10 AM to 10 PM.
# - Tell guests what time the park closes today if they ask.
# - If the park is closed, explain when it will open next.
# - Only mention hours when the guest's question is about hours or being open. Do not volunteer hours unprompted.
# </current_time_awareness>

# <pricing_clarification>
# The park has many attractions with different prices. If a guest asks a general pricing question like "How much does it cost?" or "What are your prices?" without specifying an activity, ask which activity they mean before answering. Ask one short clarifying question, like "Which activity are you asking about?" Do not guess a price. Once the attraction is known, answer using the RAG context.
# </pricing_clarification>

# <birthday_party_rules>
# 1. EXISTING BOOKINGS: If a guest asks about a party they already booked, wants to change, reschedule, update guest counts, or check booking details, immediately transfer to a human agent. Say something like "Let me connect you with our team so they can pull up your booking and help with that." Do not attempt to modify bookings.
# 2. NEW BOOKINGS: If a guest wants to book a new birthday party, first ask "Do you already know which party package you'd like to book?" If they don't know, explain the packages from the knowledge base. If they already know and want to proceed with booking, transfer to a human agent.
# </birthday_party_rules>

# <conversation_style>
# - Do NOT end responses with repetitive closing phrases like "If you'd like to book or need more details feel free to contact us" or "Please contact us for more information." Only provide information relevant to the question asked. Avoid scripted customer-service language.
# - Do NOT repeatedly say "AeroSports Scarborough" in every response. Use the park name only when necessary. Say "we" instead. For example, say "We've got trampolines, laser tag, and mini golf" not "At AeroSports Scarborough we offer trampolines, laser tag, and mini golf."
# - When asking clarifying questions, ask only ONE question per message. Do not list multiple options in a single question. Say "Which activity are you asking about?" not "Are you asking about laser tag, mini golf, trampoline passes, or birthday parties?"
# </conversation_style>

# <location>
# The park is located on Birchmount Road in Scarborough. Birchmount is part of Scarborough. Never say the Birchmount location does not exist. If asked about the location, say "We're on Birchmount Road in Scarborough."
# </location>"""

VOICE_SYSTEM_PROMPT = """\
<identity>
You are Dan, a front desk staff member at AeroSports Scarborough trampoline park, answering inbound phone calls. You are a person, not an AI. If confused, respond like Dan would: "Sorry, I didn't quite catch that — what can I help you with?"
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
Existing bookings (any mention of "my booking," "my party," "I booked," wanting to change, cancel, or reschedule): acknowledge briefly and stop. Do not ask for names, dates, or package types. Do not quote rescheduling policies. Just: "Oh okay, sure" — then wait.

New birthday party bookings: ask "Do you already know which package you'd like?" If not, explain the packages from the context. If yes, say "Perfect, let me connect you with our team to get that booked."

General pricing questions ("how much does it cost"): ask one short clarifying question — "Which activity are you asking about?" — before answering.

Height or age requirements: frame casually as safety. "Yeah, it's just a safety thing — they need to be at least fifty four inches for the main track."

Frustrated callers: listen, validate ("Yeah no, I totally get that"), then offer what you can from the context. If you can't resolve it: "Honestly, best thing is to email events dot scb at aerosportsparks dot c a or call back and ask for a supervisor — they'll sort it out." Never promise a fix you can't deliver.
</handling_specific_situations>

<location_and_hours>
We're on Birchmount Road in Scarborough. Park hours: Sunday to Thursday ten to nine, Friday and Saturday ten to ten. Only mention hours if the caller asks.
</location_and_hours>"""

_VOICE_FALLBACK = (
    "I'm having trouble answering right now. "
    "Please call us directly at 289-454-5555."
)

_END_CALL_KEYWORDS_DEFINITE = [
    "bye", "goodbye", "good bye", "that's all", "thats all", "thanks bye",
    "thank you bye", "no thanks bye", "i'm good thanks", "im good thanks",
    "that's everything", "thats everything", "i'll call back", "ill call back",
    "i'll call later", "ill call later", "have a good one", "you too bye",
]

_END_CALL_KEYWORDS_MAYBE = [
    "speak to a manager", "speak to manager", "talk to a manager", "talk to manager",
    "speak to someone", "speak to a supervisor", "talk to a supervisor",
    "human agent", "real person", "actual person", "a person",
    "my booking", "my reservation", "my party", "i already booked",
    "i booked", "my existing booking", "change my booking", "cancel my booking",
    "reschedule", "refund", "complaint", "i want to complain",
    "file a complaint", "unhappy with", "not happy with",
]

# Strict subset of booking-change triggers — high confidence that the caller
# has an existing booking they want to modify. Matching any of these bypasses
# the main LLM and starts the booking-change capture state machine.
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
]

# Messages that the LLM can answer from conversation history alone — no RAG needed.
_SKIP_RAG_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|okay|ok|no|nope|nah|thanks|thank you|"
    r"bye|goodbye|hi|hello|hey|good morning|good afternoon|good evening|"
    r"mm.?hmm|uh.?huh|go ahead|please|right|got it|sounds good|awesome|"
    r"cool|great|perfect|alright|nice|for sure|no worries)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _should_skip_rag(user_message: str, conversation_history: list[dict]) -> bool:
    """
    Decide whether the message can be answered by the LLM from conversation
    history alone, without calling the RAG API.

    Skips RAG for:
    - Pure acknowledgments/greetings with no information need ("yes", "hi", "thanks")
    - BUT only when there IS conversation history (the LLM has context to work with)

    Short follow-ups like "yes" that follow an assistant question ("wanna know
    about the premium?") ARE skipped here — the LLM already has the knowledge
    base context from the previous turn in its conversation history.
    """
    if not conversation_history:
        return False  # First message — always search
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
    """
    Use the LLM to rewrite a follow-up question into a self-contained search query.

    Uses the last 3 conversation turns (6 messages) as context.
    Falls back to the original message on any error.
    """
    if not conversation_history:
        return user_message

    recent = conversation_history[-6:]
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
            max_tokens=150,
            temperature=0.0,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        if not rewritten or len(rewritten) > 500:
            return user_message
        logger.info("Rewritten query: %r → %r", user_message[:100], rewritten[:200])
        return rewritten
    except Exception as exc:
        logger.error("Query rewrite failed, using original: %s", exc)
        return user_message


def _clean_token_for_tts(token: str) -> str:
    """
    Lightweight per-token cleaner for streaming to Twilio.

    Strips markdown symbols that TTS would read aloud (asterisks, hashes,
    backticks, dollar signs) but preserves leading/trailing whitespace so
    words don't get smashed together.
    """
    token = re.sub(r"\*+", "", token)    # bold / italic asterisks
    token = re.sub(r"#+", "", token)     # heading hashes
    token = re.sub(r"`+", "", token)     # code backticks
    token = re.sub(r"\$", "", token)     # dollar signs (prices spoken by number)
    return token


def _clean_for_tts(text: str) -> str:
    """Strip markdown, thinking blocks, and symbols that sound bad when read aloud by TTS.

    Applied to the full accumulated response (not per-token) for conversation history.
    """
    # Remove Qwen3 <think>…</think> reasoning blocks (greedy — may span many lines)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip unclosed <think> block (model still "thinking" when it hit max_tokens)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    # Strip orphaned </think> left by cancelled/interrupted streams
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"\*+", "", text)                               # bold / italic asterisks
    text = re.sub(r"#+\s*", "", text)                             # ATX headings
    text = re.sub(r"`+", "", text)                                # inline code / fences
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)        # [label](url) → label
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)  # leading bullet dashes
    # Emojis — TTS reads these as their text description ("party popper", "glowing star")
    text = re.sub(r"[\U00010000-\U0010FFFF]", "", text)          # supplementary plane (most emojis)
    text = re.sub(r"[\u2600-\u27BF\uFE00-\uFE0F]", "", text)    # misc symbols + variation selectors
    # Abbreviations → full words for natural TTS
    text = re.sub(r"\bmins?\b", "minutes", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhrs?\b", "hours", text, flags=re.IGNORECASE)
    # Dollar amounts → spoken form: "$39.90" → "39 90"
    text = re.sub(r"\$(\d+)\.(\d{2})", r"\1 \2", text)
    text = re.sub(r"\$(\d+)", r"\1 dollars", text)
    # Bare decimal prices without $ sign: "39.90" → "39 90"
    text = re.sub(r"\b(\d+)\.(\d{2})\b", r"\1 \2", text)
    text = re.sub(r"\n+", " ", text).strip()
    return text


def _check_end_keywords(user_text: str) -> str:
    """
    Fast keyword pre-filter for end-of-call detection.

    Returns:
        "definite" — user clearly ended the call, skip classifier
        "maybe"    — user may need a human, run classifier to confirm
        "none"     — normal turn, no classifier needed
    """
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
    """
    Return True if the user's message clearly indicates they want to modify
    an existing booking. Triggers the booking-change capture state machine
    in the WebSocket handler.
    """
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
    """
    Run a small LLM call to decide if this turn should end the call.

    Returns None if the call should continue.
    Returns a dict {summary, needs_human, flag_reason} if the call should end.
    """
    import json as _json

    prompt = (
        _CLASSIFIER_PROMPT
        .replace("<<USER_TEXT>>", user_text[:500])
        .replace("<<ASSISTANT_TEXT>>", assistant_text[:500])
    )

    try:
        client = _make_async_client()
        response = await client.chat.completions.create(
            model=_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=150,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        logger.info("Classifier raw output: %r", raw[:300])

        # Model sometimes wraps in markdown fences — strip them
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        # Extract the first {...} block in case the model added prose
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("Classifier returned no JSON object")
            return None

        parsed = _json.loads(match.group(0))

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
    """Build an end-call decision dict for the 'definite' keyword path (no LLM call)."""
    return {
        "summary": f"Caller ended the call after: {user_text[:100]}",
        "needs_human": False,
        "flag_reason": "",
    }


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
        # Deduplicate — the RAG API often returns the same content from
        # different source paths.  Keep the highest-scored version.
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
            "Direct the caller to phone 289-454-5555 or email events.scb@aerosportsparks.ca."
        )

    # Current time context for hours awareness
    now = datetime.now(_TORONTO_TZ)
    time_text = (
        f"CURRENT TIME: {now.strftime('%A, %B %d, %Y at %I:%M %p')} (Eastern Time)"
    )

    logger.info("VOICE RAG CONTEXT sent to LLM:\n%s", context_text)
    logger.info("VOICE TIME CONTEXT: %s", time_text)

    # Single system message — fewer messages = faster local inference
    system_content = f"{VOICE_SYSTEM_PROMPT}\n\n{time_text}\n\n{context_text}"
    messages: list[dict] = [
        {"role": "system", "content": system_content},
    ]
    # Keep last 6 turns (12 messages) to reduce context size
    messages.extend(conversation_history[-20:])
    messages.append({"role": "user", "content": user_message})
    return messages


async def _call_llm_async(messages: list[dict]) -> str:
    """Non-streaming async LLM call with a short token budget for voice."""
    client = _make_async_client()
    response = await client.chat.completions.create(
        model=_VOICE_MODEL,
        messages=messages,
        stream=False,
        max_tokens=500,
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
    t_total_start = time.perf_counter()
    user_text = user_text.strip()[:500]

    # Fetch history early so we can use it for query enrichment
    history = conversation_store.get(call_sid)

    # Lazily create a pipeline logger for this session (first turn)
    pl = _session_loggers.get(call_sid) or init_session_logger(call_sid)

    # Step 1 — incoming transcript
    pl.log_transcript(user_text)

    # Skip RAG for simple acknowledgments/greetings the LLM can handle
    # from conversation history alone (e.g. "yes", "thanks", "hi")
    t_rag_ms = 0.0
    if _should_skip_rag(user_text, history):
        logger.info("[%s] Skipping RAG — conversational message: %s", call_sid, user_text)
        rag_docs = []
        # Steps 2 & 3 — mark as skipped
        pl.log_refined_query(user_text, "__SKIPPED__")
        pl.log_rag_results([])
    else:
        search_query = await _rewrite_query(user_text, history)
        # Step 2 — refined query
        pl.log_refined_query(user_text, search_query)

        t_rag_start = time.perf_counter()
        rag_docs = await _query_rag_api(search_query, top_k=7)
        t_rag_ms = (time.perf_counter() - t_rag_start) * 1000
        logger.info("[%s] LATENCY rag_api=%.0fms  docs=%d", call_sid, t_rag_ms, len(rag_docs))
        # Step 3 — RAG results
        pl.log_rag_results(rag_docs)

    messages = _build_voice_messages(user_text, rag_docs, history)

    # Step 4 — full LLM context
    pl.log_llm_context(messages)

    logger.info("[%s] Full messages sent to LLM (%d messages):", call_sid, len(messages))
    for i, m in enumerate(messages):
        logger.info("[%s]   msg[%d] role=%s content=%.300s", call_sid, i, m["role"], m["content"])

    # Non-streaming async LLM call
    t_llm_start = time.perf_counter()
    try:
        raw_reply = await _call_llm_async(messages)
        logger.info("[%s] Raw LLM response: %s", call_sid, raw_reply)
        # Step 5 — raw LLM response
        pl.log_llm_response(raw_reply)
    except Exception as exc:
        logger.error("LLM error on voice call %s: %s", call_sid, exc)
        pl.log_error(f"LLM call failed: {exc}", exc)
        raw_reply = _VOICE_FALLBACK
    t_llm_ms = (time.perf_counter() - t_llm_start) * 1000
    logger.info("[%s] LATENCY llm_call=%.0fms", call_sid, t_llm_ms)

    reply = _clean_for_tts(raw_reply)
    logger.info("[%s] Cleaned TTS reply: %s", call_sid, reply)
    # Step 6 — final text sent to TTS / caller
    pl.log_final_response(reply)

    t_total_ms = (time.perf_counter() - t_total_start) * 1000
    logger.info("[%s] LATENCY total=%.0fms (rag=%.0fms + llm=%.0fms)", call_sid, t_total_ms, t_rag_ms, t_llm_ms)

    # Persist turn to the shared store (same TTL / trim logic applies)
    conversation_store.add(call_sid, "user", user_text)
    conversation_store.add(call_sid, "assistant", reply)

    return reply


async def prepare_voice_stream(
    call_sid: str,
    user_text: str,
):
    """
    Prepare RAG context and LLM messages for streaming voice response.

    Returns (messages, user_text_cleaned) so the caller can drive the
    streaming loop and handle cancellation.  Conversation history is
    updated with the user turn here; the assistant turn is the caller's
    responsibility (it depends on how much was actually spoken).
    """
    t_start = time.perf_counter()
    user_text = user_text.strip()[:500]

    history = conversation_store.get(call_sid)

    # Lazily create a pipeline logger for this session (first turn)
    pl = _session_loggers.get(call_sid) or init_session_logger(call_sid)

    # Step 1 — incoming transcript
    pl.log_transcript(user_text)

    # Skip RAG for simple acknowledgments/greetings the LLM can handle
    # from conversation history alone (e.g. "yes", "thanks", "hi")
    if _should_skip_rag(user_text, history):
        logger.info("[%s] Skipping RAG — conversational message: %s", call_sid, user_text)
        rag_docs = []
        # Steps 2 & 3 — mark as skipped
        pl.log_refined_query(user_text, "__SKIPPED__")
        pl.log_rag_results([])
    else:
        search_query = await _rewrite_query(user_text, history)
        # Step 2 — refined query
        pl.log_refined_query(user_text, search_query)

        t_rag_start = time.perf_counter()
        rag_docs = await _query_rag_api(search_query, top_k=7)
        t_rag_ms = (time.perf_counter() - t_rag_start) * 1000
        logger.info("[%s] LATENCY rag_api=%.0fms  docs=%d", call_sid, t_rag_ms, len(rag_docs))
        # Step 3 — RAG results
        pl.log_rag_results(rag_docs)

    messages = _build_voice_messages(user_text, rag_docs, history)

    # Step 4 — full LLM context
    pl.log_llm_context(messages)

    logger.info("[%s] Full messages sent to LLM (%d messages):", call_sid, len(messages))
    for i, m in enumerate(messages):
        logger.info("[%s]   msg[%d] role=%s content=%.300s", call_sid, i, m["role"], m["content"])

    # Record the user turn now; assistant turn is recorded by the caller
    # after streaming completes (or is interrupted).
    conversation_store.add(call_sid, "user", user_text)

    t_ms = (time.perf_counter() - t_start) * 1000
    logger.info("[%s] LATENCY prepare_voice_stream=%.0fms", call_sid, t_ms)
    return messages


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")


# async def stream_voice_tokens(messages: list[dict]):
#     """
#     Async generator that yields TTS-safe sentences from Ollama.

#     Tokens are accumulated into a buffer and flushed whenever a sentence
#     boundary (., !, ?) is detected.  Yielding whole sentences — rather than
#     individual tokens — prevents choppy audio from the TTS engine while still
#     keeping latency low (first sentence is sent as soon as it finishes).

#     Each token is lightly cleaned (_clean_token_for_tts) to strip markdown
#     symbols (**, #, `, $) while preserving whitespace so words stay separated.
#     Full cleaning (_clean_for_tts) is applied by the caller to the accumulated
#     response for conversation history.
#     """
#     client = _make_async_client()
#     stream = await client.chat.completions.create(
#         model=_VOICE_MODEL,
#         messages=messages,
#         stream=True,
#         max_tokens=500,
#         temperature=0.3,
#         top_p=0.9,
#     )

#     buffer = ""

#     async for chunk in stream:
#         if not chunk.choices:
#             continue
#         token = chunk.choices[0].delta.content
#         if not token:
#             continue

#         cleaned = _clean_token_for_tts(token)
#         if not cleaned:
#             continue

#         buffer += cleaned

#         # Yield every complete sentence as soon as we see a boundary
#         match = _SENTENCE_END_RE.search(buffer)
#         while match:
#             end_idx = match.end()
#             sentence = buffer[:end_idx].strip()
#             if sentence:
#                 yield sentence
#             buffer = buffer[end_idx:]
#             match = _SENTENCE_END_RE.search(buffer)

#     # Flush any trailing text that didn't end with punctuation
#     remainder = buffer.strip()
#     if remainder:
#         yield remainder

_SENTENCE_END_RE = re.compile(r"([.!?]+(?:\s+|$))")

# async def stream_voice_tokens(messages: list[dict]):
#     client = _make_async_client()
#     stream = await client.chat.completions.create(
#         model=_VOICE_MODEL,
#         messages=messages,
#         stream=True,
#         max_tokens=500,
#         temperature=0.3,
#         # Keep the model focused on the voice prompt
#         extra_body={"keep_alive": -1} 
#     )

#     buffer = ""

#     async for chunk in stream:
#         if not chunk.choices:
#             continue
#         token = chunk.choices[0].delta.content
#         if not token:
#             continue

#         # 1. Light clean for symbols but KEEP the punctuation
#         cleaned_token = _clean_token_for_tts(token)
#         buffer += cleaned_token

#         # 2. Extract complete sentences
#         while True:
#             # This regex looks for punctuation followed by a space OR end of string
#             match = _SENTENCE_END_RE.search(buffer)
#             if not match:
#                 break
            
#             # Extract the sentence including its punctuation
#             end_idx = match.end()
#             sentence = buffer[:end_idx].strip()
            
#             if sentence:
#                 # 3. THE DOT FIX: Ensure there is exactly ONE space after the period
#                 # and that the period is attached to the word before it.
#                 if any(c.isalnum() for c in sentence):
#                     # We add a space to force the TTS to "breathe" and recognize 
#                     # the period as a stop, not a "dot".
#                     yield f"{sentence} "
            
#             buffer = buffer[end_idx:]

#     # 4. Flush remainder
#     remainder = buffer.strip()
#     if remainder:
#         # Final safety check: if the model ended without punctuation, 
#         # add a period so the TTS doesn't sound cut off.
#         if remainder[-1] not in ".!?":
#             remainder += "."
#         yield f"{remainder} "


async def stream_voice_tokens(messages: list[dict]):
    """
    Async generator that yields TTS-safe text chunks from the LLM.

    Strategy: "fast first, smooth after"
    - First chunk: flush on comma, period, or after ~8 words — whichever
      comes first. This minimizes time-to-first-audio.
    - Subsequent chunks: flush on sentence boundaries (. ! ?) for natural
      TTS pacing.
    """
    client = _make_async_client()
    stream = await client.chat.completions.create(
        model=_VOICE_MODEL,
        messages=messages,
        stream=True,
        max_tokens=500,
        temperature=0.3,
        extra_body={"keep_alive": -1},
    )

    buffer = ""
    first_chunk_sent = False
    word_count = 0

    # Thresholds for the first flush
    FIRST_FLUSH_WORDS = 8        # flush after this many words if no punctuation yet
    FIRST_FLUSH_CHARS = 60       # safety cap — flush if buffer gets long without breaks

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
            # --- FIRST CHUNK: flush aggressively ---
            # Option A: We hit a comma or sentence-end punctuation
            early_break = re.search(r"[,.:;!?]\s", buffer)
            # Option B: We've accumulated enough words without any punctuation
            word_flush = word_count >= FIRST_FLUSH_WORDS
            # Option C: Buffer is getting long (model produced a run-on)
            char_flush = len(buffer) >= FIRST_FLUSH_CHARS

            if early_break:
                # Flush up to and including the punctuation + space
                end_idx = early_break.end()
                chunk_text = buffer[:end_idx].strip()
                buffer = buffer[end_idx:]
                word_count = buffer.count(" ")
            elif word_flush or char_flush:
                # No punctuation yet — flush everything we have
                chunk_text = buffer.strip()
                buffer = ""
                word_count = 0
            else:
                continue  # keep accumulating

            if chunk_text and any(c.isalnum() for c in chunk_text):
                yield f"{chunk_text} "
                first_chunk_sent = True

        else:
            # --- SUBSEQUENT CHUNKS: sentence boundaries ---
            while True:
                match = _SENTENCE_END_RE.search(buffer)
                if not match:
                    break

                end_idx = match.end()
                sentence = buffer[:end_idx].strip()
                buffer = buffer[end_idx:]

                if sentence and any(c.isalnum() for c in sentence):
                    yield f"{sentence} "

    # --- FLUSH REMAINDER ---
    remainder = buffer.strip()
    if remainder:
        if remainder[-1] not in ".!?":
            remainder += "."
        yield f"{remainder} "