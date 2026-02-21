"""
Builds the full messages list for Llama:

  1. System prompt  — AeroBot personality + rules
  2. RAG context    — numbered knowledge-base excerpts (second system message)
  3. History        — trimmed to MAX_CONVERSATION_TURNS
  4. User message   — the current user turn
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import SearchResult

MAX_CONVERSATION_TURNS: int = int(os.environ.get("MAX_CONVERSATION_TURNS", "10"))

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are AeroBot, the friendly and helpful customer assistant for AeroSports Scarborough trampoline park.

PERSONALITY:
- Professional yet warm and approachable
- Enthusiastic about AeroSports but never pushy
- Concise — answer in 2-4 sentences unless the question genuinely needs more detail
- Use natural language, not bullet points (unless comparing packages side by side)

RULES:
1. ONLY answer based on the provided context. If the context doesn't contain the answer, say so honestly and offer to help via phone or email.
2. NEVER invent prices, hours, policies, package names, jump pass names, or details not in the context.
3. ALL prices are + tax unless stated otherwise — always mention this.
4. When mentioning booking or purchasing, direct users to: https://www.aerosportsparks.ca
5. For complex questions (custom events, large groups, special accommodations), suggest emailing events.scb@aerosportsparks.ca or calling 289-454-5555.
6. If asked about other AeroSports locations (Oakville, London, St. Catharines), let them know you currently only have Scarborough information and suggest they call the relevant park.
7. Keep all responses focused on the Scarborough location only.

CONTACT INFO (use when relevant):
- Phone: 289-454-5555 (press 3 for the park)
- Email: events.scb@aerosportsparks.ca
- Hours: Sun–Thur 10 am–8 pm, Fri–Sat 10 am–10 pm
- Website: https://www.aerosportsparks.ca

FORMATTING:
- Few sentences depending on requirement, not bullet lists (unless comparing packages)
- Include specific prices when available
- Always add "+tax" after prices
- Bold key info like prices and times when helpful\
"""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_messages(
    user_message: str,
    rag_context: list,  # list[SearchResult]
    conversation_history: list[dict],
) -> list[dict]:
    """
    Return the complete messages list ready to send to Llama.

    Structure:
        [system: SYSTEM_PROMPT]
        [system: KNOWLEDGE BASE CONTEXT ...]
        [...trimmed conversation history...]
        [user: user_message]
    """
    # Format RAG context as numbered, labelled excerpts
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
            "Answer honestly that you don't have that specific information "
            "and direct the user to call 289-454-5555 or email events.scb@aerosportsparks.ca."
        )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context_text},
    ]

    # Trim history: keep the last MAX_CONVERSATION_TURNS complete turns
    trimmed = conversation_history[-(MAX_CONVERSATION_TURNS * 2):]
    messages.extend(trimmed)

    messages.append({"role": "user", "content": user_message})
    return messages
