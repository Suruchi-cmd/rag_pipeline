"""
Thin async client for the in-house RAG API at $RAG_API_URL.

Used by voice_handler to fetch source documents for the LLM prompt.

Returned shape (per the /rag/retrieve endpoint):
    [
      {"content": str, "score": float,
       "metadata": {"file_name": str | None,
                    "page": int | None,
                    "source": str | None}},
      ...
    ]
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")

# One AsyncClient reused across calls so we skip the TCP+TLS handshake per query.
_client = httpx.AsyncClient(timeout=15.0)


async def query_rag(query: str, top_k: int = 5) -> list[dict]:
    """POST /rag/retrieve and return the source_documents list (empty on error)."""
    try:
        resp = await _client.post(
            f"{_RAG_API_URL}/rag/retrieve",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json().get("source_documents", [])
    except Exception as exc:
        logger.error("RAG API error: %s", exc)
        return []
