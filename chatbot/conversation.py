"""
In-memory conversation store keyed by session_id.

- Sessions expire after SESSION_TIMEOUT seconds of inactivity.
- MAX_CONVERSATION_TURNS most-recent turns are retained per session.
- Async-safe via asyncio.Lock — every method is a coroutine; callers must await.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from chatbot.config import settings

SESSION_TIMEOUT: int = settings.SESSION_TIMEOUT
MAX_CONVERSATION_TURNS: int = settings.MAX_CONVERSATION_TURNS


@dataclass
class _Session:
    messages: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class ConversationStore:
    """Async-safe in-memory store for per-session message history."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> list[dict]:
        """Return a copy of the message list for *session_id* (empty list if unknown)."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_active = time.time()
            return list(session.messages)

    async def add(self, session_id: str, role: str, content: str) -> None:
        """Append a message to *session_id*, trimming history to MAX_CONVERSATION_TURNS."""
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = _Session()
            session = self._sessions[session_id]
            session.messages.append({"role": role, "content": content})
            max_msgs = MAX_CONVERSATION_TURNS * 2
            if len(session.messages) > max_msgs:
                session.messages = session.messages[-max_msgs:]
            session.last_active = time.time()

    async def replace_last_assistant(self, session_id: str, content: str) -> None:
        """Replace the most recent assistant message, or append if none exists."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            for i in range(len(session.messages) - 1, -1, -1):
                if session.messages[i]["role"] == "assistant":
                    session.messages[i]["content"] = content
                    session.last_active = time.time()
                    return
            session.messages.append({"role": "assistant", "content": content})
            session.last_active = time.time()

    async def clear(self, session_id: str) -> None:
        """Delete the session entirely."""
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def cleanup_expired(self) -> int:
        """Remove sessions idle longer than SESSION_TIMEOUT. Returns count removed."""
        cutoff = time.time() - SESSION_TIMEOUT
        async with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
            for sid in expired:
                del self._sessions[sid]
        return len(expired)


# Module-level singleton shared across the whole server process.
conversation_store = ConversationStore()
