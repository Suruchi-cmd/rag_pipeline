"""Embedding utilities: embed_text() and embed_batch() supporting Voyage AI or local models."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model/client singletons
# ---------------------------------------------------------------------------

_voyage_client = None
_local_model = None


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai

        if not config.VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY is not set in .env")
        _voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
        logger.info("Voyage AI client initialised (model=%s)", config.VOYAGE_MODEL)
    return _voyage_client


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(config.LOCAL_MODEL_NAME)
        logger.info("Local sentence-transformer loaded: %s", config.LOCAL_MODEL_NAME)
    return _local_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a list of floats."""
    return embed_batch([text])[0]


def embed_batch(texts: list[str], batch_size: int = config.EMBED_BATCH_SIZE) -> list[list[float]]:
    """
    Embed a list of strings in batches.

    Args:
        texts: Strings to embed.
        batch_size: Max texts per API/model call.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    provider = config.EMBEDDING_PROVIDER
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        logger.debug("Embedding batch %d–%d of %d", start, start + len(batch), len(texts))

        if provider == "voyage":
            embeddings = _embed_voyage(batch)
        elif provider == "local":
            embeddings = _embed_local(batch)
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}. Use 'voyage' or 'local'.")

        all_embeddings.extend(embeddings)

    return all_embeddings


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _embed_voyage(texts: list[str]) -> list[list[float]]:
    client = _get_voyage_client()
    result = client.embed(texts, model=config.VOYAGE_MODEL, input_type="document")
    return [list(vec) for vec in result.embeddings]


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _get_local_model()
    vectors: np.ndarray = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()
