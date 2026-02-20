"""
ingest.py — Initial load of aerosports_scb_knowledge_base.json into pgvector.

Steps:
1. Load & validate 93 chunks from JSON
2. Embed each chunk (question + "\n" + answer) in batches
3. Upsert into knowledge_chunks (ON CONFLICT DO UPDATE)

Usage:
    python ingest.py
    python ingest.py --dry-run       # validate only, no DB writes
    python ingest.py --path data/other.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

import config
import embedding as emb
from models import ChunkRecord

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_JSON_PATH = Path("data/aerosports_scb_knowledge_base.json")

UPSERT_SQL = """
INSERT INTO knowledge_chunks
    (id, category, subcategory, location, question, answer, tags, embedding)
VALUES
    (%(id)s, %(category)s, %(subcategory)s, %(location)s,
     %(question)s, %(answer)s, %(tags)s, %(embedding)s::vector)
ON CONFLICT (id) DO UPDATE SET
    category    = EXCLUDED.category,
    subcategory = EXCLUDED.subcategory,
    location    = EXCLUDED.location,
    question    = EXCLUDED.question,
    answer      = EXCLUDED.answer,
    tags        = EXCLUDED.tags,
    embedding   = EXCLUDED.embedding,
    updated_at  = CURRENT_TIMESTAMP;
"""


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------


def load_chunks(json_path: Path) -> list[ChunkRecord]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_chunks: list[dict] = data.get("chunks", [])
    declared_total: int = data.get("metadata", {}).get("total_chunks", len(raw_chunks))

    logger.info("JSON declares %d chunks; found %d in file", declared_total, len(raw_chunks))

    chunks: list[ChunkRecord] = []
    errors: list[str] = []

    for item in raw_chunks:
        missing = [k for k in ("id", "category", "subcategory", "location", "question", "answer", "tags") if k not in item]
        if missing:
            errors.append(f"Chunk {item.get('id', '?')} missing fields: {missing}")
            continue

        chunks.append(
            ChunkRecord(
                id=item["id"],
                category=item["category"],
                subcategory=item["subcategory"],
                location=item["location"],
                question=item["question"],
                answer=item["answer"],
                tags=item["tags"],
            )
        )

    if errors:
        for e in errors:
            logger.error("Validation error: %s", e)
        logger.warning("%d chunks skipped due to validation errors", len(errors))

    logger.info("Loaded %d valid chunks", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


def embed_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """Embed all chunks in batches; mutates and returns the list."""
    texts = [c.embed_text() for c in chunks]
    logger.info("Embedding %d chunks with provider=%r …", len(chunks), config.EMBEDDING_PROVIDER)

    vectors: list[list[float]] = []
    batch_size = config.EMBED_BATCH_SIZE

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding batches", unit="batch"):
        batch = texts[start : start + batch_size]
        vectors.extend(emb.embed_batch(batch, batch_size=batch_size))

    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec

    logger.info("Embeddings complete (dim=%d)", len(vectors[0]) if vectors else 0)
    return chunks


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def upsert_chunks(chunks: list[ChunkRecord]) -> None:
    pool = config.get_db_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                for chunk in tqdm(chunks, desc="Upserting chunks", unit="chunk"):
                    cur.execute(
                        UPSERT_SQL,
                        {
                            "id": chunk.id,
                            "category": chunk.category,
                            "subcategory": chunk.subcategory,
                            "location": chunk.location,
                            "question": chunk.question,
                            "answer": chunk.answer,
                            "tags": chunk.tags,
                            "embedding": chunk.embedding,
                        },
                    )
        logger.info("Upserted %d chunks into knowledge_chunks", len(chunks))
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge base JSON into pgvector")
    parser.add_argument("--path", type=Path, default=DEFAULT_JSON_PATH, help="Path to JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; skip embedding + DB write")
    args = parser.parse_args()

    if not args.path.exists():
        logger.error("File not found: %s", args.path)
        sys.exit(1)

    t0 = time.monotonic()
    chunks = load_chunks(args.path)

    if args.dry_run:
        logger.info("Dry run — skipping embedding and DB upsert.")
        return

    chunks = embed_chunks(chunks)
    upsert_chunks(chunks)
    config.close_db_pool()

    elapsed = time.monotonic() - t0
    logger.info("Ingest complete in %.1fs — %d chunks stored.", elapsed, len(chunks))


if __name__ == "__main__":
    main()
