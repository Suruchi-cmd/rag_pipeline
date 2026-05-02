from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Call(SQLModel, table=True):
    __tablename__ = "calls"

    id: Optional[int] = Field(default=None, primary_key=True)
    call_sid: str = Field(index=True, unique=True)
    phone_number: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    status: str = Field(default="active")  # active | completed | abandoned
    summary: Optional[str] = None
    needs_human: bool = Field(default=False)
    flag_reason: Optional[str] = None
    total_turns: int = Field(default=0)
    avg_turn_ms: Optional[float] = Field(default=None)


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: int = Field(foreign_key="calls.id", index=True)
    role: str  # user | assistant
    content: str
    turn_number: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    was_interrupted: bool = Field(default=False)


class RAGRetrieval(SQLModel, table=True):
    __tablename__ = "rag_retrievals"

    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: int = Field(foreign_key="calls.id", index=True)
    turn_number: int
    original_query: str
    rewritten_query: str
    retrieved_chunks: str = Field(default="[]")  # JSON array
    was_skipped: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BookingChange(SQLModel, table=True):
    __tablename__ = "booking_changes"

    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: int = Field(foreign_key="calls.id", index=True, unique=True)
    caller_name: str
    caller_phone: str
    change_details: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Category(SQLModel, table=True):
    __tablename__ = "categories"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CallClassification(SQLModel, table=True):
    __tablename__ = "call_classifications"

    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: int = Field(foreign_key="calls.id", index=True)
    category_id: int = Field(foreign_key="categories.id", index=True)
    classified_at: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeChunk(SQLModel, table=True):
    __tablename__ = "knowledge_chunks"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Prompt(SQLModel, table=True):
    __tablename__ = "prompts"

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(index=True, unique=True)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PromptVersion(SQLModel, table=True):
    __tablename__ = "prompt_versions"

    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_id: int = Field(foreign_key="prompts.id", index=True)
    version_no: int
    label: Optional[str] = Field(default=None)
    content: str
    is_active: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
