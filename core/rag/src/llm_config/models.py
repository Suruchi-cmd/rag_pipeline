"""Database model for runtime-mutable chat LLM configuration.

Single-row table (id=1) - LLMConfigService enforces the singleton.
"""

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class LLMConfig(SQLModel, table=True):
    """Chat LLM settings editable at runtime via /settings/llm."""

    id: int | None = Field(default=1, primary_key=True)
    ollama_base_url: str = Field(...)
    chat_model: str = Field(...)
    request_timeout: int = Field(default=120)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
