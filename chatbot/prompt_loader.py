"""
Cache-aware reader for active prompt content.

Voice handlers call ``get_prompt(slug)`` on every turn. To avoid hitting the
DB each time, the value is cached in process memory. The prompts router calls
``invalidate_prompt(slug)`` (or ``invalidate_all()``) whenever an active
version changes so the next read refreshes from DB.
"""

from __future__ import annotations

import logging
import threading

from sqlmodel import Session

from chatbot.prompt_defaults import (
    CLASSIFIER_PROMPT_DEFAULT,
    CLASSIFIER_PROMPT_SLUG,
    REWRITE_PROMPT_DEFAULT,
    REWRITE_PROMPT_SLUG,
    VOICE_SYSTEM_PROMPT_DEFAULT,
    VOICE_SYSTEM_PROMPT_SLUG,
)
from database.repository import get_active_prompt_content
from database.session import engine

logger = logging.getLogger(__name__)

_DEFAULTS = {
    VOICE_SYSTEM_PROMPT_SLUG: VOICE_SYSTEM_PROMPT_DEFAULT,
    CLASSIFIER_PROMPT_SLUG: CLASSIFIER_PROMPT_DEFAULT,
    REWRITE_PROMPT_SLUG: REWRITE_PROMPT_DEFAULT,
}

_cache: dict[str, str] = {}
_lock = threading.Lock()


def get_prompt(slug: str) -> str:
    cached = _cache.get(slug)
    if cached is not None:
        return cached
    with _lock:
        cached = _cache.get(slug)
        if cached is not None:
            return cached
        try:
            with Session(engine) as session:
                content = get_active_prompt_content(session, slug)
        except Exception as exc:
            logger.error("Failed to load prompt %r from DB: %s", slug, exc)
            content = None
        if content is None:
            content = _DEFAULTS.get(slug, "")
            if not content:
                logger.warning("No prompt found for slug %r — returning empty string", slug)
        _cache[slug] = content
        return content


def invalidate_prompt(slug: str) -> None:
    with _lock:
        _cache.pop(slug, None)


def invalidate_all() -> None:
    with _lock:
        _cache.clear()
