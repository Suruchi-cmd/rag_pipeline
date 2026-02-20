"""
sync.py — Google Sheets Change Log → pgvector sync.

Flow:
1. Read version cell (M1) from Change Log sheet
2. Compare to sync_state.last_version in DB → skip if unchanged
3. Read Change Log rows where synced = FALSE
4. For each row, process UPDATE / ADD / DELETE:
   - chunk_id is read directly from the Change Log row (column D)
   - UPDATE: fetch latest Q+A from the source sheet, re-embed, UPDATE DB row
   - ADD:    build new ChunkRecord from change data, embed, INSERT
   - DELETE: DELETE FROM knowledge_chunks WHERE id = chunk_id
5. Mark rows synced=TRUE, synced_at=now in the sheet
6. Update sync_state.last_version + last_synced_at in DB
7. Append to sync_history

Usage:
    python sync.py              # normal run
    python sync.py --force      # ignore version check, process all unsynced rows
    python sync.py --dry-run    # simulate, no DB or sheet writes
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

import psycopg2

import config
import embedding as emb
from models import ChangeLogEntry, ChunkRecord

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Column indices in the Change Log sheet (0-based)
COL_CHANGE_ID    = 0   # A
COL_TIMESTAMP    = 1   # B
COL_SHEET_NAME   = 2   # C
COL_CHUNK_ID     = 3   # D
COL_CHANGE_TYPE  = 4   # E
COL_FIELD        = 5   # F
COL_OLD_VALUE    = 6   # G
COL_NEW_VALUE    = 7   # H
COL_SYNCED       = 8   # I
COL_SYNCED_AT    = 9   # J

HEADER_ROW = 0  # row index of header (0-based); data starts at row 1


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _get_last_version(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT last_version FROM sync_state WHERE id = 1")
        row = cur.fetchone()
        return row[0] if row else "1.0"


def _update_sync_state(conn, version: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE sync_state
            SET last_version = %s, last_synced_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (version,),
        )


def _log_sync_history(conn, entry: ChangeLogEntry) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sync_history (change_id, chunk_id, change_type, field_changed)
            VALUES (%s, %s, %s, %s)
            """,
            (entry.change_id, entry.chunk_id, entry.change_type, entry.field_changed),
        )


def _upsert_chunk(conn, chunk: ChunkRecord) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
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
                updated_at  = CURRENT_TIMESTAMP
            """,
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


def _delete_chunk(conn, chunk_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM knowledge_chunks WHERE id = %s", (chunk_id,))
        logger.info("Deleted chunk %r (%d row affected)", chunk_id, cur.rowcount)


def _fetch_existing_chunk(conn, chunk_id: str) -> Optional[ChunkRecord]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, category, subcategory, location, question, answer, tags FROM knowledge_chunks WHERE id = %s",
            (chunk_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return ChunkRecord(
        id=row[0], category=row[1], subcategory=row[2],
        location=row[3], question=row[4], answer=row[5], tags=list(row[6]),
    )


# ---------------------------------------------------------------------------
# Sheet helpers
# ---------------------------------------------------------------------------


def _read_version(spreadsheet) -> str:
    sheet = spreadsheet.worksheet(config.CHANGE_LOG_SHEET)
    return sheet.acell(config.VERSION_CELL).value or "1.0"


def _read_unsynced_rows(spreadsheet) -> tuple[list[ChangeLogEntry], list[int]]:
    """
    Returns (entries, sheet_row_indices) where sheet_row_indices are 1-based
    row numbers in the Google Sheet (for marking synced later).
    """
    sheet = spreadsheet.worksheet(config.CHANGE_LOG_SHEET)
    all_rows = sheet.get_all_values()

    entries: list[ChangeLogEntry] = []
    row_indices: list[int] = []

    for i, row in enumerate(all_rows):
        if i == HEADER_ROW:
            continue
        # Pad short rows
        row = row + [""] * (COL_SYNCED_AT + 1 - len(row))

        synced_flag = row[COL_SYNCED].strip().upper()
        if synced_flag in ("TRUE", "YES", "1", "DONE"):
            continue

        chunk_id = row[COL_CHUNK_ID].strip()
        change_type = row[COL_CHANGE_TYPE].strip().upper()
        if not chunk_id or not change_type:
            continue

        entries.append(
            ChangeLogEntry(
                change_id=row[COL_CHANGE_ID].strip(),
                timestamp=row[COL_TIMESTAMP].strip(),
                sheet_name=row[COL_SHEET_NAME].strip(),
                chunk_id=chunk_id,
                change_type=change_type,
                field_changed=row[COL_FIELD].strip(),
                old_value=row[COL_OLD_VALUE].strip(),
                new_value=row[COL_NEW_VALUE].strip(),
            )
        )
        row_indices.append(i + 1)  # sheet rows are 1-based

    return entries, row_indices


def _mark_synced(spreadsheet, row_index: int) -> None:
    """Set synced=TRUE and synced_at=now for a sheet row."""
    sheet = spreadsheet.worksheet(config.CHANGE_LOG_SHEET)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sheet.update_cell(row_index, COL_SYNCED + 1, "TRUE")
    sheet.update_cell(row_index, COL_SYNCED_AT + 1, now_str)


# ---------------------------------------------------------------------------
# Sheet data fetchers for UPDATE / ADD
# ---------------------------------------------------------------------------


def _fetch_chunk_from_faqs_sheet(spreadsheet, chunk_id: str) -> Optional[ChunkRecord]:
    """
    Read the FAQs sheet and rebuild a ChunkRecord for the given chunk_id.
    FAQs sheet columns: chunk_id, category, question, answer
    """
    try:
        sheet = spreadsheet.worksheet("FAQs")
    except Exception:
        logger.error("FAQs sheet not found in spreadsheet")
        return None

    rows = sheet.get_all_values()
    for row in rows[1:]:  # skip header
        if not row:
            continue
        row = row + [""] * 4
        if row[0].strip() == chunk_id:
            return ChunkRecord(
                id=chunk_id,
                category=row[1].strip() or "FAQ",
                subcategory=row[1].strip() or "General",
                location="Scarborough",
                question=row[2].strip(),
                answer=row[3].strip(),
                tags=[],
            )
    return None


def _fetch_chunk_from_source_sheet(spreadsheet, entry: ChangeLogEntry, existing: Optional[ChunkRecord]) -> Optional[ChunkRecord]:
    """
    Rebuild a ChunkRecord from the source sheet.

    Strategy:
    - FAQs → read directly from FAQs sheet (Q+A are full columns)
    - Everything else → apply the field change to the existing chunk (simple field patch)
    """
    chunk_id = entry.chunk_id

    if entry.sheet_name == "FAQs" or chunk_id.startswith("scb_faq_"):
        return _fetch_chunk_from_faqs_sheet(spreadsheet, chunk_id)

    # For non-FAQ sheets, patch the affected field on the existing record
    if existing is None:
        logger.warning("No existing chunk found for %r — cannot patch for UPDATE", chunk_id)
        return None

    field = entry.field_changed.lower()
    new_val = entry.new_value

    chunk = ChunkRecord(
        id=existing.id,
        category=existing.category,
        subcategory=existing.subcategory,
        location=existing.location,
        question=existing.question,
        answer=existing.answer,
        tags=list(existing.tags),
    )

    if field == "answer":
        chunk.answer = new_val
    elif field == "question":
        chunk.question = new_val
    elif field == "category":
        chunk.category = new_val
    elif field == "subcategory":
        chunk.subcategory = new_val
    elif field == "tags":
        chunk.tags = [t.strip() for t in new_val.split(",") if t.strip()]
    else:
        # Unknown field — update answer as a safe fallback
        logger.warning("Unknown field %r — updating answer field as fallback", field)
        chunk.answer = new_val

    return chunk


def _build_new_chunk_from_entry(entry: ChangeLogEntry) -> Optional[ChunkRecord]:
    """Build a minimal ChunkRecord for an ADD entry using new_value as the answer."""
    if not entry.new_value:
        logger.error("ADD entry %r has no new_value — cannot create chunk", entry.change_id)
        return None

    # Infer category from chunk_id prefix
    prefix = entry.chunk_id.split("_")[1] if "_" in entry.chunk_id else "general"
    category_map = {
        "contact": "Contact",
        "hours": "Contact",
        "jump": "Pricing",
        "socks": "Pricing",
        "gokart": "Attractions",
        "glow": "Events",
        "toddler": "Events",
        "special": "Pricing",
        "bday": "Birthday Parties",
        "group": "Group Bookings",
        "facility": "Group Bookings",
        "room": "Group Bookings",
        "camp": "Aero Camp",
        "passes": "Passes",
        "faq": "FAQ",
    }
    category = category_map.get(prefix, entry.sheet_name or "General")

    return ChunkRecord(
        id=entry.chunk_id,
        category=category,
        subcategory=entry.field_changed or "General",
        location="Scarborough",
        question=entry.field_changed or f"Information about {entry.chunk_id}",
        answer=entry.new_value,
        tags=[],
    )


# ---------------------------------------------------------------------------
# Per-entry processors
# ---------------------------------------------------------------------------


def _process_update(conn, spreadsheet, entry: ChangeLogEntry, dry_run: bool) -> bool:
    existing = _fetch_existing_chunk(conn, entry.chunk_id)
    chunk = _fetch_chunk_from_source_sheet(spreadsheet, entry, existing)

    if chunk is None:
        logger.error("Skipping UPDATE %r — could not build updated chunk", entry.chunk_id)
        return False

    if not dry_run:
        chunk.embedding = emb.embed_text(chunk.embed_text())
        _upsert_chunk(conn, chunk)
        _log_sync_history(conn, entry)

    logger.info("UPDATE %r (field=%r)", entry.chunk_id, entry.field_changed)
    return True


def _process_add(conn, spreadsheet, entry: ChangeLogEntry, dry_run: bool) -> bool:
    chunk = _build_new_chunk_from_entry(entry)

    if chunk is None:
        return False

    if not dry_run:
        chunk.embedding = emb.embed_text(chunk.embed_text())
        _upsert_chunk(conn, chunk)
        _log_sync_history(conn, entry)

    logger.info("ADD %r", entry.chunk_id)
    return True


def _process_delete(conn, entry: ChangeLogEntry, dry_run: bool) -> bool:
    if not dry_run:
        _delete_chunk(conn, entry.chunk_id)
        _log_sync_history(conn, entry)

    logger.info("DELETE %r", entry.chunk_id)
    return True


# ---------------------------------------------------------------------------
# Main sync orchestration
# ---------------------------------------------------------------------------


def sync(force: bool = False, dry_run: bool = False) -> None:
    if dry_run:
        logger.info("DRY RUN — no DB or sheet writes will occur")

    spreadsheet = config.get_spreadsheet()
    sheet_version = _read_version(spreadsheet)
    logger.info("Sheet version: %s", sheet_version)

    db_pool = config.get_db_pool()
    conn = db_pool.getconn()

    try:
        db_version = _get_last_version(conn)
        logger.info("DB version:    %s", db_version)

        if not force and sheet_version == db_version:
            logger.info("Versions match — nothing to sync. Exiting.")
            return

        entries, row_indices = _read_unsynced_rows(spreadsheet)
        logger.info("Found %d unsynced change log entries", len(entries))

        if not entries:
            if not dry_run:
                with conn:
                    _update_sync_state(conn, sheet_version)
            logger.info("No unsynced rows. Updated version to %s.", sheet_version)
            return

        success_count = 0
        for entry, sheet_row in zip(entries, row_indices):
            try:
                with conn:  # transaction per change
                    if entry.change_type == "UPDATE":
                        ok = _process_update(conn, spreadsheet, entry, dry_run)
                    elif entry.change_type == "ADD":
                        ok = _process_add(conn, spreadsheet, entry, dry_run)
                    elif entry.change_type == "DELETE":
                        ok = _process_delete(conn, entry, dry_run)
                    else:
                        logger.warning("Unknown change_type %r for %r — skipping", entry.change_type, entry.chunk_id)
                        ok = False

                if ok and not dry_run:
                    _mark_synced(spreadsheet, sheet_row)
                    success_count += 1

            except Exception as exc:
                logger.error("Failed to process change %r: %s", entry.change_id, exc, exc_info=True)
                # Transaction rolled back automatically by context manager

        if not dry_run:
            with conn:
                _update_sync_state(conn, sheet_version)

        logger.info(
            "Sync complete — %d/%d changes applied. DB version now %s.",
            success_count,
            len(entries),
            sheet_version,
        )

    finally:
        db_pool.putconn(conn)
        config.close_db_pool()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Google Sheets Change Log to pgvector")
    parser.add_argument("--force", action="store_true", help="Ignore version check")
    parser.add_argument("--dry-run", action="store_true", help="Simulate; no writes")
    args = parser.parse_args()
    sync(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
