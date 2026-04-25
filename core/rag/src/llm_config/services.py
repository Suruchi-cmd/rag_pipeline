"""Service layer for LLM config."""

import logging

from src.config import settings

from .models import LLMConfig
from .repositories import LLMConfigRepository

logger = logging.getLogger(__name__)


class LLMConfigService:
    def __init__(self, repository: LLMConfigRepository) -> None:
        self.repository = repository

    def get(self) -> LLMConfig:
        row = self.repository.get()
        if not row:
            row = self.bootstrap()
        return row

    def bootstrap(self) -> LLMConfig:
        return self.repository.seed_if_missing(
            ollama_base_url=settings.DEFAULT_OLLAMA_BASE_URL,
            chat_model=settings.DEFAULT_CHAT_MODEL,
            request_timeout=settings.DEFAULT_REQUEST_TIMEOUT,
        )

    def update(
        self,
        ollama_base_url: str | None = None,
        chat_model: str | None = None,
        request_timeout: int | None = None,
    ) -> LLMConfig:
        return self.repository.update(
            ollama_base_url=ollama_base_url,
            chat_model=chat_model,
            request_timeout=request_timeout,
        )
