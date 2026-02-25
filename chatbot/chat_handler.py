"""
Chat handler — the main orchestration pipeline.

For each user message:
  1. Run semantic + hybrid search against pgvector (in thread-pool to avoid blocking).
  2. Deduplicate and rank results.
  3. Build the full Llama message list via prompt_builder.
  4. Stream tokens from the HuggingFace API (sync generator bridged to async via threading).
  5. Optionally append fallback CTAs.
  6. Persist the exchange in the conversation store.
  7. Yield a final "done" event carrying source metadata.

Yields dicts:
    {"type": "token",  "content": str}   — one per streamed token
    {"type": "done",   "sources": list}  — one final event
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from typing import AsyncGenerator

# ---------------------------------------------------------------------------
# Ensure the repo root is importable so we can reach the flat RAG modules
# (config.py, search.py, models.py, embedding.py).
# When uvicorn is run from the repo root this is already on sys.path, but
# this guard makes the import work regardless of launch directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from search import hybrid_search, semantic_search  # noqa: E402
from models import SearchResult  # noqa: E402

from chatbot.conversation import conversation_store  # noqa: E402
from chatbot.fallback import detect_fallback  # noqa: E402
from chatbot.llm import _FALLBACK_MSG, generate_response  # noqa: E402
from chatbot.prompt_builder import build_messages  # noqa: E402

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHUNKS: int = int(os.environ.get("MAX_CONTEXT_CHUNKS", "5"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deduplicate(results: list[SearchResult], max_k: int) -> list[SearchResult]:
    """Merge result lists, keeping only the first occurrence of each chunk id."""
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if r.chunk.id not in seen:
            seen.add(r.chunk.id)
            out.append(r)
        if len(out) >= max_k:
            break
    return out


def _format_sources(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "id": r.chunk.id,
            "category": r.chunk.category,
            "subcategory": r.chunk.subcategory,
            "question": r.chunk.question,
            "score": round(r.similarity_score, 3),
        }
        for r in results
    ]


async def _stream_llm_async(messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    Bridge the synchronous HuggingFace streaming generator to an async generator.

    Runs the blocking HF call in a daemon thread and pushes tokens into an
    asyncio.Queue that the async generator awaits.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    exc_holder: list[Exception] = []

    def _run() -> None:
        try:
            for token in generate_response(messages):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:  # noqa: BLE001
            exc_holder.append(exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=_run, daemon=True).start()

    while True:
        token = await queue.get()
        if token is None:
            if exc_holder:
                raise exc_holder[0]
            return
        yield token


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def handle_message(
    session_id: str,
    user_message: str,
) -> AsyncGenerator[dict, None]:
    """
    Full RAG → LLM → response pipeline as an async generator.

    Yields:
        {"type": "token",  "content": "<text>"}  for each streamed token
        {"type": "done",   "sources": [...]}      as the final event
    """
    # Sanitise input
    user_message = user_message.strip()[:500]

    # ------------------------------------------------------------------
    # 1. Retrieve context — run both searches concurrently in thread-pool
    # ------------------------------------------------------------------
    try:
        semantic_results, hybrid_results = await asyncio.gather(
            asyncio.to_thread(semantic_search, user_message, None, 5),
            asyncio.to_thread(hybrid_search, user_message, None, 3),
        )
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        semantic_results, hybrid_results = [], []

    merged = _deduplicate(semantic_results + hybrid_results, max_k=MAX_CONTEXT_CHUNKS)

    # ------------------------------------------------------------------
    # 2. Build prompt
    # ------------------------------------------------------------------
    history = conversation_store.get(session_id)
    messages = build_messages(user_message, merged, history)

    logger.info(
        "[%s] PROMPT MESSAGES (%d):\n%s",
        session_id,
        len(messages),
        "\n".join(
            f"  [{m['role'].upper()}] {m['content'][:300]}{'...' if len(m['content']) > 300 else ''}"
            for m in messages
        ),
    )

    # ------------------------------------------------------------------
    # 3. Stream from Llama
    # ------------------------------------------------------------------
    full_response = ""
    try:
        async for token in _stream_llm_async(messages):
            full_response += token
            yield {"type": "token", "content": token}
    except Exception as exc:
        logger.error("LLM streaming failed: %s", exc)
        yield {"type": "token", "content": _FALLBACK_MSG}
        full_response = _FALLBACK_MSG

    # ------------------------------------------------------------------
    # 4. Fallback CTAs (appended after LLM response if needed)
    # ------------------------------------------------------------------
    cta = detect_fallback(user_message, full_response, merged)
    if cta:
        yield {"type": "token", "content": cta}
        full_response += cta

    # ------------------------------------------------------------------
    # 5. Persist conversation turn
    # ------------------------------------------------------------------
    logger.info("[%s] RESPONSE: %s", session_id, full_response)

    conversation_store.add(session_id, "user", user_message)
    conversation_store.add(session_id, "assistant", full_response)

    # ------------------------------------------------------------------
    # 6. Final event with sources
    # ------------------------------------------------------------------
    yield {"type": "done", "sources": _format_sources(merged)}
