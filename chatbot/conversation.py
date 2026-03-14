"""
In-memory conversation store keyed by session_id.

- Sessions expire after SESSION_TIMEOUT seconds of inactivity.
- MAX_CONVERSATION_TURNS most-recent turns are retained per session.
- Thread-safe with a Lock (FastAPI runs async but sync code runs in threads).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

SESSION_TIMEOUT: int = 30 * 60  # 30 minutes
MAX_CONVERSATION_TURNS: int = int(os.environ.get("MAX_CONVERSATION_TURNS", "30"))


@dataclass
class _Session:
    messages: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class ConversationStore:
    """Thread-safe in-memory store for per-session message history."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[dict]:
        """Return a copy of the message list for *session_id* (empty list if unknown)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_active = time.time()
            return list(session.messages)

    def add(self, session_id: str, role: str, content: str) -> None:
        """Append a message to *session_id*, trimming history to MAX_CONVERSATION_TURNS."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = _Session()
            session = self._sessions[session_id]
            session.messages.append({"role": role, "content": content})
            # Keep only the last N complete turns (each turn = 1 user + 1 assistant msg)
            max_msgs = MAX_CONVERSATION_TURNS * 2
            if len(session.messages) > max_msgs:
                session.messages = session.messages[-max_msgs:]
            session.last_active = time.time()

    def clear(self, session_id: str) -> None:
        """Delete the session entirely."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        """Remove sessions that have been idle longer than SESSION_TIMEOUT. Returns count removed."""
        cutoff = time.time() - SESSION_TIMEOUT
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
            for sid in expired:
                del self._sessions[sid]
        return len(expired)


# Module-level singleton shared across the whole server process.
conversation_store = ConversationStore()
