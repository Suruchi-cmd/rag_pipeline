"""Repository for LLM config persistence."""

import logging
from datetime import datetime, timezone

from sqlmodel import Session, create_engine, select

from src.config import settings

from .models import LLMConfig

logger = logging.getLogger(__name__)


class LLMConfigRepository:
    SINGLETON_ID = 1

    def __init__(self) -> None:
        self.engine = create_engine(settings.database_url)

    def get(self) -> LLMConfig | None:
        with Session(self.engine) as session:
            return session.exec(
                select(LLMConfig).where(LLMConfig.id == self.SINGLETON_ID)
            ).first()

    def seed_if_missing(
        self, ollama_base_url: str, chat_model: str, request_timeout: int
    ) -> LLMConfig:
        with Session(self.engine) as session:
            existing = session.exec(
                select(LLMConfig).where(LLMConfig.id == self.SINGLETON_ID)
            ).first()
            if existing:
                return existing
            row = LLMConfig(
                id=self.SINGLETON_ID,
                ollama_base_url=ollama_base_url,
                chat_model=chat_model,
                request_timeout=request_timeout,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            logger.info("Seeded LLMConfig: %s @ %s", chat_model, ollama_base_url)
            return row

    def update(
        self,
        ollama_base_url: str | None = None,
        chat_model: str | None = None,
        request_timeout: int | None = None,
    ) -> LLMConfig:
        with Session(self.engine) as session:
            row = session.exec(
                select(LLMConfig).where(LLMConfig.id == self.SINGLETON_ID)
            ).first()
            if not row:
                raise ValueError("LLMConfig not seeded")
            if ollama_base_url is not None:
                row.ollama_base_url = ollama_base_url
            if chat_model is not None:
                row.chat_model = chat_model
            if request_timeout is not None:
                row.request_timeout = request_timeout
            row.updated_at = datetime.now(timezone.utc)
            session.add(row)
            session.commit()
            session.refresh(row)
            return row
