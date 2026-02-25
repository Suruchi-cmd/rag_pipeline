"""
sync.py — Google Sheets Change Log → pgvector sync.

Flow:
1. Read version cell (M1) from Change Log sheet
2. Compare to sync_state.last_version in DB → skip if unchanged
3. Read Change Log rows where synced = FALSE
4. Group rows by chunk_id (avoid re-embedding same chunk multiple times)
5. For each chunk group:
   - DELETE: remove from DB
   - UPDATE/ADD on FAQs: re-fetch full Q+A from FAQs sheet, re-embed, upsert
   - UPDATE/ADD on data sheets: re-fetch ALL rows for that chunk_id from
     source sheet, rebuild structured text, re-embed, upsert
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
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import psycopg2

import config
import embedding as emb
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

# Maps chunk_id prefix → category for non-FAQ chunks
CATEGORY_MAP = {
    "contact":  "Contact",
    "hours":    "Contact",
    "jump":     "Pricing",
    "socks":    "Pricing",
    "gokart":   "Attractions",
    "glow":     "Events",
    "toddler":  "Events",
    "special":  "Accessibility",
    "bday":     "Birthday Parties",
    "group":    "Group Bookings",
    "facility": "Group Bookings",
    "room":     "Group Bookings",
    "camp":     "Aero Camp",
    "passes":   "Passes",
    "faq":      "FAQ",
}

# Maps chunk_id prefix → natural-language question for non-FAQ chunks
QUESTION_MAP = {
    "contact":  "How do I contact AeroSports Scarborough?",
    "hours":    "What are the hours for AeroSports Scarborough?",
    "jump":     "What are the jump prices at AeroSports Scarborough?",
    "socks":    "Do I need special socks for jumping?",
    "gokart":   "What are the go karting options and prices?",
    "glow":     "What is Glow at AeroSports?",
    "toddler":  "What is Toddler Time at AeroSports?",
    "special":  "What are the special needs accommodations?",
    "bday":     "What are the birthday party packages?",
    "group":    "How do group bookings work?",
    "facility": "What does a facility rental include?",
    "room":     "What are the party room options?",
    "camp":     "What is Aero Camp?",
    "passes":   "What passes are available?",
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
                "id":          chunk.id,
                "category":    chunk.category,
                "subcategory": chunk.subcategory,
                "location":    chunk.location,
                "question":    chunk.question,
                "answer":      chunk.answer,
                "tags":        chunk.tags,
                "embedding":   chunk.embedding,
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

    # Build batch updates: [(row, col, value), ...]
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
# Chunk builders
# ---------------------------------------------------------------------------


def _get_chunk_prefix(chunk_id: str) -> str:
    """Extract the prefix from a chunk_id: scb_jump_003 → jump"""
    parts = chunk_id.split("_")
    return parts[1] if len(parts) >= 2 else "general"


def _fetch_chunk_from_faqs_sheet(spreadsheet, chunk_id: str) -> Optional[ChunkRecord]:
    """
    Read the FAQs sheet and rebuild a ChunkRecord for the given chunk_id.
    FAQs columns: chunk_id | category | question | answer
    """
    try:
        sheet = spreadsheet.worksheet("FAQs")
    except Exception:
        logger.error("FAQs sheet not found")
        return None

    rows = sheet.get_all_values()
    for row in rows[1:]:
        if not row:
            continue
        row = row + [""] * 4
        if row[0].strip() == chunk_id:
            category = row[1].strip() or "FAQ"
            question = row[2].strip()
            answer = row[3].strip()

            if not question and not answer:
                logger.warning("FAQ %r has no question or answer", chunk_id)
                return None

            return ChunkRecord(
                id=chunk_id,
                category=category,
                subcategory=category,
                location="Scarborough",
                question=question,
                answer=answer,
                tags=["faq", category.lower().replace(" ", "_")],
            )
    logger.warning("FAQ %r not found in FAQs sheet", chunk_id)
    return None


def _fetch_chunk_from_data_sheet(
    spreadsheet, sheet_name: str, chunk_id: str
) -> Optional[ChunkRecord]:
    """
    Re-reads the source sheet, finds ALL rows for the given chunk_id,
    and rebuilds the chunk as structured text.

    Handles sub-tables (Birthday Parties, Group Bookings, Aero Camp)
    by detecting rows where col A = "chunk_id" as sub-table headers.
    """
    try:
        sheet = spreadsheet.worksheet(sheet_name)
    except Exception:
        logger.error("Sheet %r not found", sheet_name)
        return None

    all_rows = sheet.get_all_values()
    if not all_rows:
        return None

    # ── Collect all rows belonging to this chunk_id ──
    current_headers = all_rows[0]
    chunk_rows: list[dict[str, str]] = []

    for i, row in enumerate(all_rows):
        if i == 0:
            continue

        # Detect sub-table header rows
        if row and row[0].strip().lower() == "chunk_id":
            current_headers = row
            continue

        row_chunk_id = row[0].strip() if row else ""
        if not row_chunk_id or row_chunk_id != chunk_id:
            continue

        # Build header → value dict for this row
        row_dict = {}
        for col_idx, header in enumerate(current_headers):
            header_clean = header.strip()
            if not header_clean or header_clean.lower() == "chunk_id":
                continue
            val = row[col_idx].strip() if col_idx < len(row) else ""
            if val:
                row_dict[header_clean] = val
        if row_dict:
            chunk_rows.append(row_dict)

    if not chunk_rows:
        logger.warning("No rows found for %r in %r", chunk_id, sheet_name)
        return None

    # ── Build structured text ──
    text_parts = []
    for row_dict in chunk_rows:
        parts = [f"{k}: {v}" for k, v in row_dict.items()]
        text_parts.append(" | ".join(parts))

    answer_text = "\n".join(text_parts)

    prefix = _get_chunk_prefix(chunk_id)
    category = CATEGORY_MAP.get(prefix, sheet_name or "General")
    question = QUESTION_MAP.get(prefix, f"Information about {chunk_id}")

    return ChunkRecord(
        id=chunk_id,
        category=category,
        subcategory=prefix,
        location="Scarborough",
        question=question,
        answer=answer_text,
        tags=[prefix, sheet_name.lower().replace(" ", "_")],
    )


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
    2. Re-fetch the entire chunk from the source sheet
    3. Re-embed once
    4. Upsert once

    This means if 5 fields changed on the same chunk, we do 1 fetch + 1 embed
    instead of 5.
    """

    # Check if any entry in the group is a DELETE
    has_delete = any(e.change_type == "DELETE" for e in entries)

    if has_delete:
        if not dry_run:
            with conn:
                _delete_chunk(conn, chunk_id)
                for entry in entries:
                    _log_sync_history(conn, entry)
        logger.info("DELETE %r (%d change log rows)", chunk_id, len(entries))
        return True

    # Determine source sheet from the entries
    # All entries for the same chunk_id should come from the same sheet,
    # but take the most recent one to be safe
    last_entry = entries[-1]
    sheet_name = last_entry.sheet_name

    # ── Fetch the full chunk from source ──
    if sheet_name == "FAQs" or chunk_id.startswith("scb_faq_"):
        chunk = _fetch_chunk_from_faqs_sheet(spreadsheet, chunk_id)
    else:
        chunk = _fetch_chunk_from_data_sheet(spreadsheet, sheet_name, chunk_id)

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

        # ── Group entries by chunk_id ──
        # Preserves order: processes chunks in the order of their first appearance
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
