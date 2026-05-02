"""
Call classification using the local LLM.

classify_and_store(call_id) — fetch transcript + categories, classify, persist.
Called automatically when a call ends and triggered manually via /api/categories/resync.

Uses the same OpenAI-compat async client as the voice pipeline (chatbot/llm.py)
so there are no extra DB engines or LangChain chains created per classification.
"""

from __future__ import annotations

import json
import logging
import re

from config import settings

logger = logging.getLogger(__name__)

_PROMPT = """\
You are a call classifier for AeroSports Scarborough, a trampoline and entertainment park.

Read the transcript below and select ALL categories that this call is about.

Available categories:
{categories}

Rules:
- Return ONLY a valid JSON array of category names, spelled exactly as listed above
- A call can belong to multiple categories
- Only pick categories that were clearly discussed
- Use "General Inquiries" when the topic is general or does not match other categories
- If truly nothing fits, return []
- No explanation, no markdown — JSON array only

Transcript (Caller / Maya):
{transcript}

JSON:"""


async def classify_call(
    messages: list[dict],
    category_names: list[str],
) -> list[str]:
    """
    Run LLM classification. Returns list of matching category names.

    Reuses the voice pipeline's async OpenAI-compat client — no extra
    SQLAlchemy engines or LangChain chains are created.
    """
    if not category_names or not messages:
        return []

    transcript_lines = [
        f"{'Caller' if m['role'] == 'user' else 'Maya'}: {m['content'][:300]}"
        for m in messages
        if m.get("content", "").strip()
    ]
    transcript = "\n".join(transcript_lines[:80])
    categories_str = "\n".join(f"- {n}" for n in category_names)
    prompt = _PROMPT.format(categories=categories_str, transcript=transcript)

    try:
        from chatbot.llm import _make_async_client

        client = _make_async_client()
        response = await client.chat.completions.create(
            model=settings.VOICE_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=150,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            logger.warning("Classifier returned no JSON array: %r", raw[:300])
            return []
        parsed: list = json.loads(match.group(0))
        valid = [n for n in parsed if isinstance(n, str) and n in category_names]
        logger.info("Call classified → %s", valid)
        return valid
    except Exception as exc:
        logger.error("Classification error: %s", exc)
        return []


async def classify_and_store(call_id: int) -> list[str]:
    """Fetch transcript + categories from DB, classify, persist results.

    Returns the list of matched category names (empty if nothing matched
    or the call had no transcript).
    """
    from database.models import Category
    from database.repository import get_messages, list_categories, upsert_call_classifications
    from database.session import engine
    from sqlmodel import Session

    with Session(engine) as session:
        messages_rows = get_messages(session, call_id)
        categories: list[Category] = list_categories(session)

    if not messages_rows or not categories:
        logger.debug("classify_and_store: nothing to classify for call %d", call_id)
        return []

    msg_dicts = [{"role": m.role, "content": m.content} for m in messages_rows]
    category_names = [c.name for c in categories]

    matched_names = await classify_call(msg_dicts, category_names)
    matched_ids = [c.id for c in categories if c.name in matched_names]

    with Session(engine) as session:
        upsert_call_classifications(session, call_id, matched_ids)

    logger.info("Call %d → %d categories: %s", call_id, len(matched_ids), matched_names)
    return matched_names
