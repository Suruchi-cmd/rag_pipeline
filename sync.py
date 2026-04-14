"""
sync.py — Google Sheets Change Log → pgvector sync.

Flow:
1. Read version cell (M1) from Change Log sheet
2. Compare to sync_state.last_version in DB → skip if unchanged
3. Read Change Log rows where synced = FALSE
4. Group rows by chunk_id (avoid re-embedding same chunk multiple times)
5. For each chunk group:
   - DELETE: remove from DB
   - Promotions status change to non-Active: DELETE the chunk
   - UPDATE/ADD: re-read source sheet, rebuild via chunk_builder, re-embed, upsert
6. Mark all processed rows synced=TRUE, synced_at=now in the sheet
7. Update sync_state.last_version + last_synced_at in DB
8. Append to sync_history

Usage:
    python sync.py              # normal run
    python sync.py --force      # ignore version check, process all unsynced rows
    python sync.py --dry-run    # simulate, no DB or sheet writes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import psycopg2

import config
import embedding as emb
from chunk_builder import build_chunks_from_sheet
from models import ChangeLogEntry, ChunkRecord

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Column indices in the Change Log sheet (0-based)
COL_CHANGE_ID  = 0   # A
COL_TIMESTAMP  = 1   # B
COL_SHEET_NAME = 2   # C
COL_CHUNK_ID   = 3   # D
COL_CHANGE_TYPE = 4  # E
COL_FIELD      = 5   # F
COL_OLD_VALUE  = 6   # G
COL_NEW_VALUE  = 7   # H
COL_SYNCED     = 8   # I
COL_SYNCED_AT  = 9   # J

HEADER_ROW = 0

# Maps chunk_id prefix → category
CATEGORY_MAP = {
    "contact":   "Contact",
    "hours":     "Contact",
    "links":     "Contact",
    "jump":      "Pricing",
    "socks":     "Pricing",
    "gokart":    "Go Karting",
    "glow":      "Special Programs",
    "toddler":   "Special Programs",
    "special":   "Special Programs",
    "attr":      "Attractions",
    "bday":      "Birthday Parties",
    "group":     "Group Bookings",
    "corporate": "Group Bookings",
    "school":    "Group Bookings",
    "fundraise": "Group Bookings",
    "facility":  "Group Bookings",
    "rooms":     "Group Bookings",
    "camp":      "Aero Camp",
    "passes":    "Passes",
    "promo":     "Promotions",
    "rules":     "Park Rules",
    "faq":       "FAQ",
    "voice":     "Voice Scripts",
    "qr":        "Quick Replies",
}

# Maps chunk_id prefix → natural-language question
QUESTION_MAP = {
    "contact":   "How do I contact AeroSports Scarborough?",
    "hours":     "What are the hours for AeroSports Scarborough?",
    "links":     "What are the booking and waiver links?",
    "jump":      "What are the jump prices at AeroSports Scarborough?",
    "socks":     "Do I need special socks for jumping?",
    "gokart":    "What are the go karting options and prices?",
    "glow":      "What is Glow at AeroSports?",
    "toddler":   "What is Toddler Time at AeroSports?",
    "special":   "What are the special programs at AeroSports?",
    "attr":      "What attractions are available at AeroSports Scarborough?",
    "bday":      "What are the birthday party packages?",
    "group":     "How do group bookings work?",
    "corporate": "How do corporate events work?",
    "school":    "How do school field trips work?",
    "fundraise": "How do fundraising events work?",
    "facility":  "What does a facility rental include?",
    "rooms":     "What are the party room options?",
    "camp":      "What is Aero Camp?",
    "passes":    "What passes are available?",
    "promo":     "Are there any current promotions or deals?",
    "rules":     "What are the park rules?",
    "faq":       "Frequently asked questions",
    "voice":     "Information for phone callers",
    "qr":        "Quick response for common questions",
}


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
                updated_at  = CURRENT_TIMESTAMP
            """,
            {
                "id":          chunk.id,
                "category":    chunk.category,
                "subcategory": chunk.subcategory,
                "location":    chunk.location,
                "question":    chunk.question,
                "answer":      chunk.answer,
                "tags":        chunk.tags,
                "embedding":   chunk.embedding,
                "sheet_name":  chunk.sheet_name,
                "source":      chunk.source,
                "metadata":    json.dumps(chunk.metadata),
            },
        )


def _delete_chunk(conn, chunk_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM knowledge_chunks WHERE id = %s", (chunk_id,))
        logger.info("Deleted chunk %r (%d row affected)", chunk_id, cur.rowcount)


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
        row_indices.append(i + 1)

    return entries, row_indices


def _mark_synced(spreadsheet, row_index: int) -> None:
    """Set synced=TRUE and synced_at=now for a sheet row."""
    sheet = spreadsheet.worksheet(config.CHANGE_LOG_SHEET)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sheet.update_cell(row_index, COL_SYNCED + 1, "TRUE")
    sheet.update_cell(row_index, COL_SYNCED_AT + 1, now_str)


def _mark_synced_batch(spreadsheet, row_indices: list[int]) -> None:
    """Batch-mark multiple rows as synced (fewer API calls)."""
    sheet = spreadsheet.worksheet(config.CHANGE_LOG_SHEET)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    cells_to_update = []
    for row_index in row_indices:
        cells_to_update.append({
            "range": f"I{row_index}",
            "values": [["TRUE"]],
        })
        cells_to_update.append({
            "range": f"J{row_index}",
            "values": [[now_str]],
        })

    if cells_to_update:
        sheet.batch_update(cells_to_update)


# ---------------------------------------------------------------------------
# Chunk rebuilding via chunk_builder
# ---------------------------------------------------------------------------


def _get_chunk_prefix(chunk_id: str) -> str:
    """Extract the prefix from a chunk_id: scb_jump_003 → jump"""
    parts = chunk_id.split("_")
    return parts[1] if len(parts) >= 2 else "general"


def _rebuild_chunk_from_sheet(
    spreadsheet, sheet_name: str, chunk_id: str
) -> Optional[ChunkRecord]:
    """Re-read the source sheet and rebuild the specific chunk via chunk_builder."""
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except Exception:
        logger.error("Sheet %r not found", sheet_name)
        return None

    rows = worksheet.get_all_records()
    all_chunks = build_chunks_from_sheet(sheet_name, rows)
    for chunk in all_chunks:
        if chunk.id == chunk_id:
            return chunk

    logger.warning("Chunk %r not found after rebuilding sheet %r", chunk_id, sheet_name)
    return None


# ---------------------------------------------------------------------------
# Group-based processing
# ---------------------------------------------------------------------------


def _resolve_chunk_group(
    conn, spreadsheet, chunk_id: str, entries: list[ChangeLogEntry], dry_run: bool
) -> bool:
    """
    Process a group of change log entries that all share the same chunk_id.

    Instead of patching field-by-field, we:
    1. Determine the final action (DELETE wins, otherwise rebuild)
    2. Re-fetch the entire chunk from the source sheet via chunk_builder
    3. Re-embed once
    4. Upsert once
    """

    # Check if any entry is a DELETE
    has_delete = any(e.change_type == "DELETE" for e in entries)

    # Handle Promotions status changes: non-Active → DELETE
    if not has_delete:
        for e in entries:
            if (e.field_changed.lower() == "status"
                    and e.new_value.lower() != "active"
                    and _get_chunk_prefix(chunk_id) == "promo"):
                has_delete = True
                logger.info("Promotion %r status changed to %r — treating as DELETE", chunk_id, e.new_value)
                break

    if has_delete:
        if not dry_run:
            with conn:
                _delete_chunk(conn, chunk_id)
                for entry in entries:
                    _log_sync_history(conn, entry)
        logger.info("DELETE %r (%d change log rows)", chunk_id, len(entries))
        return True

    # Determine source sheet from entries
    last_entry = entries[-1]
    sheet_name = last_entry.sheet_name

    # Rebuild the chunk using chunk_builder (same logic as ingest)
    chunk = _rebuild_chunk_from_sheet(spreadsheet, sheet_name, chunk_id)

    if chunk is None:
        logger.error(
            "Could not rebuild chunk %r from sheet %r — skipping %d entries",
            chunk_id, sheet_name, len(entries),
        )
        return False

    if not dry_run:
        chunk.embedding = emb.embed_text(chunk.embed_text())
        with conn:
            _upsert_chunk(conn, chunk)
            for entry in entries:
                _log_sync_history(conn, entry)

    fields_changed = list({e.field_changed for e in entries})
    change_types = list({e.change_type for e in entries})
    logger.info(
        "UPSERT %r (types=%s, fields=%s, %d log rows)",
        chunk_id, change_types, fields_changed, len(entries),
    )
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

        # Group entries by chunk_id
        grouped: dict[str, list[tuple[ChangeLogEntry, int]]] = defaultdict(list)
        for entry, sheet_row in zip(entries, row_indices):
            grouped[entry.chunk_id].append((entry, sheet_row))

        logger.info(
            "Grouped into %d unique chunks from %d change log rows",
            len(grouped), len(entries),
        )

        success_chunks = 0
        success_rows = 0
        failed_chunks = 0

        for chunk_id, group in grouped.items():
            group_entries = [e for e, _ in group]
            group_row_indices = [r for _, r in group]

            try:
                ok = _resolve_chunk_group(conn, spreadsheet, chunk_id, group_entries, dry_run)

                if ok and not dry_run:
                    _mark_synced_batch(spreadsheet, group_row_indices)
                    success_chunks += 1
                    success_rows += len(group_row_indices)
                elif not ok:
                    failed_chunks += 1

            except Exception as exc:
                logger.error(
                    "Failed to process chunk %r (%d entries): %s",
                    chunk_id, len(group_entries), exc, exc_info=True,
                )
                failed_chunks += 1

        if not dry_run:
            with conn:
                _update_sync_state(conn, sheet_version)

        logger.info(
            "Sync complete — %d chunks (%d rows) synced, %d chunks failed. "
            "DB version now %s.",
            success_chunks, success_rows, failed_chunks, sheet_version,
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
