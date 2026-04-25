"""
Thin async client for the in-house RAG API at $RAG_API_URL.

Used by voice_handler to fetch source documents for the LLM prompt.

One AsyncClient is reused process-wide so we skip TCP+TLS handshake per query
and so connection-pool limits are explicit (no silent throttling under load).

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

import httpx

from chatbot.config import settings

logger = logging.getLogger(__name__)

_RAG_API_URL = settings.RAG_API_URL.rstrip("/")

# Sized for ~50 concurrent voice calls. RAG retrieve takes ~100-500ms each, so
# 100 connections + 50 keepalive comfortably absorbs bursts. Bump if you scale.
_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=50)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=settings.RAG_HTTP_TIMEOUT, limits=_LIMITS)
    return _client


async def close_rag_client() -> None:
    """Close the shared client. Call from FastAPI lifespan shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def query_rag(query: str, top_k: int = 5) -> list[dict]:
    """POST /rag/retrieve and return the source_documents list (empty on error)."""
    try:
        resp = await _get_client().post(
            f"{_RAG_API_URL}/rag/retrieve",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json().get("source_documents", [])
    except Exception as exc:
        logger.error("RAG API error: %s", exc)
        return []
