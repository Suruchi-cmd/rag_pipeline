"""Schemas for LLM config endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field


class LLMConfigResponse(BaseModel):
    ollama_base_url: str
    chat_model: str
    request_timeout: int
    updated_at: datetime

    class Config:
        from_attributes = True


class LLMConfigUpdateRequest(BaseModel):
    ollama_base_url: str | None = Field(default=None)
    chat_model: str | None = Field(default=None)
    request_timeout: int | None = Field(default=None, ge=1, le=600)
