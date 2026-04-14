"""
PipelineLogger — per-session structured file logger for the voice pipeline.

Each incoming call creates a new log file:
    logs/voice_session_YYYY-MM-DD_HH-MM-SS.log

Each pipeline step is written as a labelled block so the full turn can be
traced end-to-end from raw transcript to final TTS output.
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime

# Resolve logs/ relative to the repo root (two levels up from utils/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_REPO_ROOT, os.environ.get("SESSION_LOG_DIR", "logs"))

_SEP = "\u2500" * 44  # ────────────────────────────────────────────


class PipelineLogger:
    """
    Structured file logger for a single voice session.

    Instantiate once when the call arrives, call the step methods at each
    stage of the pipeline, then call ``close()`` when the session ends.

    Turn counter increments on each ``log_transcript()`` call so every
    other step in that utterance shares the same Turn # in its header.
    """

    def __init__(self, call_sid: str) -> None:
        self.call_sid = call_sid
        self._turn: int = 0

        os.makedirs(_LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"voice_session_{ts}.log"
        self._path = os.path.join(_LOG_DIR, filename)
        # line-buffered (buffering=1) so each write is flushed immediately
        self._fh = open(self._path, "w", encoding="utf-8", buffering=1)

        self._fh.write("# Voice Session Pipeline Log\n")
        self._fh.write(f"# call_sid : {call_sid}\n")
        self._fh.write(
            f"# started  : {datetime.now().isoformat(timespec='milliseconds')}\n"
        )
        self._fh.write(f"# log_file : {self._path}\n")
        self._fh.flush()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, step_name: str, content: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        header = f"[{ts}] [{step_name}] [Turn #{self._turn}]"
        self._fh.write(f"\n{header}\n{_SEP}\n{content}\n{_SEP}\n")
        self._fh.flush()

    # ------------------------------------------------------------------
    # Pipeline step methods
    # ------------------------------------------------------------------

    def log_transcript(self, text: str) -> None:
        """Step 1 — raw STT text received from ConversationRelay."""
        self._turn += 1
        self._write("INCOMING_TRANSCRIPT", text)

    def log_refined_query(self, original: str, refined: str) -> None:
        """Step 2 — query after rewriting (may equal original if unchanged)."""
        skipped = " (RAG skipped)" if refined == "__SKIPPED__" else ""
        if refined == "__SKIPPED__":
            content = f"Original : {original}\nRefined  : (not run — conversational message){skipped}"
        elif original == refined:
            content = f"Original : {original}\nRefined  : (unchanged)"
        else:
            content = f"Original : {original}\nRefined  : {refined}"
        self._write("REFINED_QUERY", content)

    def log_rag_results(self, docs: list[dict]) -> None:
        """Step 3 — chunks returned from pgvector with scores and content."""
        if not docs:
            self._write("RAG_RESULTS", "(no documents retrieved)")
            return
        lines = [f"Retrieved {len(docs)} document(s):"]
        for i, doc in enumerate(docs, 1):
            score = doc.get("score", "N/A")
            meta = doc.get("metadata", {})
            source = meta.get("file_name", meta.get("source", "unknown"))
            body = doc.get("content", "").strip()
            snippet = body[:500] + ("…" if len(body) > 500 else "")
            lines.append(f"\n[{i}]  score={score}  source={source}")
            lines.append(snippet)
        self._write("RAG_RESULTS", "\n".join(lines))

    def log_llm_context(self, messages: list[dict]) -> None:
        """Step 4 — full messages array assembled for the LLM call."""
        lines = [f"{len(messages)} message(s):"]
        for i, m in enumerate(messages):
            role = m.get("role", "?")
            body = m.get("content", "")
            lines.append(f"\n--- msg[{i}]  role={role} ---")
            lines.append(body)
        self._write("LLM_CONTEXT", "\n".join(lines))

    def log_llm_response(self, text: str) -> None:
        """Step 5 — full raw text returned by the LLM (before TTS cleaning)."""
        self._write("LLM_RAW_RESPONSE", text)

    def log_final_response(self, text: str) -> None:
        """Step 6 — TTS-cleaned text sent back to ConversationRelay / caller."""
        self._write("FINAL_RESPONSE_TO_CALLER", text)

    def log_error(self, message: str, exc: BaseException | None = None) -> None:
        """Log an error with optional full stack trace."""
        content = message
        if exc is not None:
            content += "\n\n" + traceback.format_exc()
        self._write("ERROR", content)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the log file."""
        if not self._fh.closed:
            ts = datetime.now().isoformat(timespec="milliseconds")
            self._fh.write(f"\n# session ended: {ts}\n")
            self._fh.flush()
            self._fh.close()
