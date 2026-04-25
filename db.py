import json
import sqlite3


def init_db(db_path: str = "calls.db") -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                transcript_json TEXT,
                summary TEXT,
                needs_human BOOLEAN DEFAULT 0,
                flag_reason TEXT,
                booking_change_json TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id INTEGER,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            )
        """)


def start_call(phone_number: str, db_path: str = "calls.db") -> int:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.execute(
            "INSERT INTO calls (phone_number) VALUES (?)",
            (phone_number,)
        )
        return cursor.lastrowid


def log_message(call_id: int, role: str, content: str, db_path: str = "calls.db") -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO messages (call_id, role, content) VALUES (?, ?, ?)",
            (call_id, role, content)
        )


def end_call(
    call_id: int,
    summary: str,
    needs_human: bool,
    flag_reason: str | None,
    db_path: str = "calls.db"
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE call_id = ? ORDER BY id",
            (call_id,)
        ).fetchall()
        transcript = [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in rows
        ]
        conn.execute(
            """
            UPDATE calls
            SET ended_at = CURRENT_TIMESTAMP,
                summary = ?,
                needs_human = ?,
                flag_reason = ?,
                transcript_json = ?
            WHERE id = ?
            """,
            (summary, int(needs_human), flag_reason, json.dumps(transcript), call_id)
        )


def get_call(call_id: int, db_path: str = "calls.db") -> dict | None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
        if row is None:
            return None
        return dict(row)


def save_booking_change(
    call_id: int,
    name: str,
    phone: str,
    details: str,
    db_path: str = "calls.db",
) -> None:
    """
    Save captured booking change details to the calls row and flag for human follow-up.

    Stores the three fields as a JSON object in booking_change_json and sets
    needs_human=1 with a flag_reason indicating a booking change request.
    """
    payload = json.dumps({"name": name, "phone": phone, "details": details})
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "UPDATE calls SET booking_change_json = ?, needs_human = 1, "
            "flag_reason = COALESCE(flag_reason, 'booking change requested') "
            "WHERE id = ?",
            (payload, call_id),
        )
