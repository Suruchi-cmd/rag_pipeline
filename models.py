"""Dataclasses for the AeroSports RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkRecord:
    id: str                         # e.g. "scb_faq_140"
    category: str
    subcategory: str
    location: str                   # always "Scarborough" for now
    question: str
    answer: str
    tags: list[str]
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def embed_text(self) -> str:
        """Text used as embedding input: question + newline + answer."""
        return f"{self.question}\n{self.answer}"


@dataclass
class SearchResult:
    chunk: ChunkRecord
    similarity_score: float

    def __repr__(self) -> str:
        return (
            f"SearchResult(id={self.chunk.id!r}, "
            f"score={self.similarity_score:.4f}, "
            f"question={self.chunk.question[:60]!r})"
        )


@dataclass
class ChangeLogEntry:
    change_id: str
    timestamp: str
    sheet_name: str
    chunk_id: str       # Direct reference — no mapping needed
    change_type: str    # UPDATE, ADD, DELETE
    field_changed: str
    old_value: str
    new_value: str
