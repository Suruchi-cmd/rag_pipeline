"""
ingest.py — Sheet-first ingestion: Google Sheets → chunk → JSON backup → pgvector.

Steps:
1. Read ALL sheets from the Google Sheets knowledge base
2. Chunk each sheet using voice-optimized builders (chunk_builder.py)
3. Save data/knowledge_base.json as a local backup (no embeddings)
4. Embed all chunks using BAAI/bge-m3 (1024-dim)
5. Clean-slate upsert into pgvector

Usage:
    python ingest.py                  # full ingest from Google Sheets
    python ingest.py --dry-run        # read sheets + chunk, skip embed + DB
    python ingest.py --from-json      # fallback: load from local JSON backup
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
from chunk_builder import build_chunks_from_sheet
from models import ChunkRecord

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

JSON_BACKUP_PATH = Path("data/knowledge_base.json")

SKIP_SHEETS = {'Change Log', 'Unanswered Questions'}

UPSERT_SQL = """
INSERT INTO knowledge_chunks
    (id, category, subcategory, location, question, answer, tags, embedding,
     sheet_name, source, metadata)
VALUES
    (%(id)s, %(category)s, %(subcategory)s, %(location)s,
     %(question)s, %(answer)s, %(tags)s, %(embedding)s::vector,
     %(sheet_name)s, %(source)s, %(metadata)s::jsonb)
ON CONFLICT (id) DO UPDATE SET
    category    = EXCLUDED.category,
    subcategory = EXCLUDED.subcategory,
    location    = EXCLUDED.location,
    question    = EXCLUDED.question,
    answer      = EXCLUDED.answer,
    tags        = EXCLUDED.tags,
    embedding   = EXCLUDED.embedding,
    sheet_name  = EXCLUDED.sheet_name,
    source      = EXCLUDED.source,
    metadata    = EXCLUDED.metadata,
    updated_at  = CURRENT_TIMESTAMP;
"""


# ---------------------------------------------------------------------------
# Read from Google Sheets
# ---------------------------------------------------------------------------


def read_all_sheets() -> list[ChunkRecord]:
    """Read every sheet from the Google Sheets KB and build chunks."""
    spreadsheet = config.get_spreadsheet()
    all_chunks: list[ChunkRecord] = []

    for worksheet in spreadsheet.worksheets():
        title = worksheet.title
        if title in SKIP_SHEETS:
            logger.info("Skipping sheet: %s", title)
            continue

        rows = worksheet.get_all_records()
        if not rows:
            logger.info("Empty sheet: %s", title)
            continue

        chunks = build_chunks_from_sheet(title, rows)
        all_chunks.extend(chunks)
        logger.info("%s: %d chunks", title, len(chunks))

    logger.info("Total chunks from all sheets: %d", len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# JSON backup — save & load
# ---------------------------------------------------------------------------


def save_knowledge_base_json(chunks: list[ChunkRecord], path: Path) -> None:
    """Write chunks (without embeddings) as a JSON backup file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for c in chunks:
        records.append({
            "id": c.id,
            "category": c.category,
            "subcategory": c.subcategory,
            "location": c.location,
            "question": c.question,
            "answer": c.answer,
            "tags": c.tags,
            "sheet_name": c.sheet_name,
            "source": c.source,
            "metadata": c.metadata,
        })

    data = {
        "metadata": {
            "total_chunks": len(records),
            "generated_by": "ingest.py (sheet-first)",
        },
        "chunks": records,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved JSON backup: %s (%d chunks)", path, len(records))


def load_chunks_from_json(json_path: Path) -> list[ChunkRecord]:
    """Load chunks from the local JSON backup (fallback when Sheets is unavailable)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_chunks: list[dict] = data.get("chunks", [])
    logger.info("JSON backup has %d chunks", len(raw_chunks))

    chunks: list[ChunkRecord] = []
    for item in raw_chunks:
        chunks.append(
            ChunkRecord(
                id=item["id"],
                category=item.get("category", ""),
                subcategory=item.get("subcategory", ""),
                location=item.get("location", "Scarborough"),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                tags=item.get("tags", []),
                sheet_name=item.get("sheet_name", ""),
                source=item.get("source", "knowledge_base"),
                metadata=item.get("metadata", {}),
            )
        )
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
# Upsert (clean slate)
# ---------------------------------------------------------------------------


def upsert_chunks(chunks: list[ChunkRecord]) -> None:
    """Delete all existing chunks and insert fresh ones."""
    pool = config.get_db_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM knowledge_chunks")
                logger.info("Cleared knowledge_chunks table")
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
                            "sheet_name": chunk.sheet_name,
                            "source": chunk.source,
                            "metadata": json.dumps(chunk.metadata),
                        },
                    )
        logger.info("Upserted %d chunks into knowledge_chunks", len(chunks))
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def full_ingest(dry_run: bool = False) -> None:
    """Read all sheets from Google Sheets, chunk, embed, and load into pgvector."""
    t0 = time.monotonic()

    # 1. Read all sheets and build chunks
    all_chunks = read_all_sheets()

    if not all_chunks:
        logger.error("No chunks produced from any sheet — aborting")
        return

    # 2. Save as JSON backup (no embeddings)
    save_knowledge_base_json(all_chunks, JSON_BACKUP_PATH)

    if dry_run:
        logger.info("Dry run — skipping embedding and DB upsert.")
        return

    # 3. Embed all chunks
    all_chunks = embed_chunks(all_chunks)

    # 4. Clean slate + upsert into pgvector
    upsert_chunks(all_chunks)

    elapsed = time.monotonic() - t0
    logger.info("Ingest complete in %.1fs — %d chunks stored.", elapsed, len(all_chunks))


def json_ingest(json_path: Path, dry_run: bool = False) -> None:
    """Fallback: load from local JSON backup and ingest."""
    t0 = time.monotonic()
    chunks = load_chunks_from_json(json_path)

    if dry_run:
        logger.info("Dry run — skipping embedding and DB upsert.")
        return

    chunks = embed_chunks(chunks)
    upsert_chunks(chunks)
    config.close_db_pool()

    elapsed = time.monotonic() - t0
    logger.info("JSON ingest complete in %.1fs — %d chunks stored.", elapsed, len(chunks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge base into pgvector")
    parser.add_argument("--dry-run", action="store_true", help="Read + chunk only; skip embedding + DB write")
    parser.add_argument("--from-json", action="store_true", help="Load from local JSON backup instead of Google Sheets")
    parser.add_argument("--json-path", type=Path, default=JSON_BACKUP_PATH, help="Path to JSON backup file")
    args = parser.parse_args()

    if args.from_json:
        if not args.json_path.exists():
            logger.error("JSON backup not found: %s", args.json_path)
            sys.exit(1)
        json_ingest(args.json_path, dry_run=args.dry_run)
    else:
        full_ingest(dry_run=args.dry_run)

    config.close_db_pool()


if __name__ == "__main__":
    main()
