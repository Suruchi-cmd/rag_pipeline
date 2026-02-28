"""
Fallback CTA detection.

After the LLM generates a response, detect_fallback() checks whether a helpful
call-to-action should be appended (booking link, phone number, email, etc.).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from models import SearchResult

# ---------------------------------------------------------------------------
# Contact constants
# ---------------------------------------------------------------------------

BOOKING_URL = "https://www.aerosportsparks.ca"
PHONE = "289-454-5555"
EMAIL = "events.scb@aerosportsparks.ca"

# Similarity threshold below which we consider the RAG context insufficient.
LOW_SIMILARITY_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Intent patterns (compiled once at import time)
# ---------------------------------------------------------------------------

_BOOKING_RE = re.compile(
    r"\b(book(?:ing)?|purchase|buy|reserve|ticket|admission|sign[\s-]?up|register)\b",
    re.IGNORECASE,
)
_CUSTOM_EVENT_RE = re.compile(
    r"\b(custom|corporate|large\s+group|special\s+(event|request|accommodat)|private\s+(party|event|session))\b",
    re.IGNORECASE,
)
_OTHER_LOCATION_RE = re.compile(
    r"\b(oakville|london|st\.?\s*catharines|other\s+location|another\s+location|different\s+location)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_fallback(
    user_message: str,
    llm_response: str,
    rag_results: list,
) -> Optional[str]:
    """
    Return an optional CTA string to append to *llm_response*, or None.

    Rules (evaluated in priority order):
    1. No RAG results, or top result below LOW_SIMILARITY_THRESHOLD
       → generic "couldn't find info" message with phone + email.
    2. Booking intent → append booking link.
    3. Custom / large-group event → append email + phone.
    4. Question about another AeroSports location → redirect note.
    """
    # --- Rule 1: no or low-confidence RAG results ---
    if not rag_results or rag_results[0].similarity_score < LOW_SIMILARITY_THRESHOLD:
        return (
            "\n\nI wasn't able to find specific information about that in my knowledge base. "
            f"For accurate details please call us at **{PHONE}** (press 3 for the park) "
            f"or email **{EMAIL}** — we're happy to help!"
        )

    ctas: list[str] = []

    # --- Rule 2: booking intent ---
    if _BOOKING_RE.search(user_message):
        ctas.append(
            f"If you're ready to book, head to our website {BOOKING_URL} to secure your spot!"
        )

    # --- Rule 3: custom / large-group events ---
    if _CUSTOM_EVENT_RE.search(user_message):
        ctas.append(
            f"For custom arrangements please email **{EMAIL}** "
            f"or call **{PHONE}** and our events team will take care of you."
        )

    # --- Rule 4: other location ---
    if _OTHER_LOCATION_RE.search(user_message):
        ctas.append(
            f"I only have information for the Scarborough location — "
            f"for other parks please visit [{BOOKING_URL}]({BOOKING_URL}) "
            f"or call the specific park directly."
        )

    if ctas:
        return "\n\n" + "  \n".join(ctas)

    return None
