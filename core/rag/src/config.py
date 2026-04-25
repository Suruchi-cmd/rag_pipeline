"""Configuration module for the Ultimate Advisor application.

Static infra/embedding settings live here. Chat LLM settings
(ollama_base_url, chat_model, request_timeout) are runtime-mutable and
stored in the llm_config DB table - see src/llm_config/.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Static configuration loaded from environment variables."""

    PG_HOST: str = Field(default="localhost")
    PG_PORT: int = Field(default=5433)
    PG_USER: str = Field(...)
    PG_PASSWORD: str = Field(...)
    PG_DATABASE: str = Field(...)

    VECTOR_TABLE_NAME: str = Field(default="documents")
    EMBEDDING_MODEL: str = Field(default="embeddinggemma")
    EMBED_DIM: int = Field(default=768)

    DEFAULT_OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    DEFAULT_CHAT_MODEL: str = Field(default="gemma3:12b")
    DEFAULT_REQUEST_TIMEOUT: int = Field(default=120)

    AUTO_INGEST: bool = Field(default=True)

    # Uvicorn server bindings (used when running `python main.py` directly).
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    RELOAD: bool = Field(default=False)
    WORKERS: int = Field(default=1)
    LOG_LEVEL: str = Field(default="info")

    # RAG ingestion consumes the LLM-enriched, voicebot-friendly markdown
    # rather than the raw parser output. The enrichment pipeline lives in
    # `parser/src/enricher.py` and is run via `run_parsers.py --enrich`.
    # Raw parser output stays at `parser/output/` (source of truth) and the
    # mirror of originals for diffing lives at `parser/output/original/`.
    DATA_FOLDER: Path = BASE_DIR / "parser/output/enriched"

    # Raw (un-enriched) markdown sits at the parser/output root. The
    # `/rag/resync-raw` endpoint re-indexes from here, skipping the
    # `enriched/` and `original/` subdirs.
    RAW_DATA_FOLDER: Path = BASE_DIR / "parser/output"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DATABASE}"

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("DATA_FOLDER", "RAW_DATA_FOLDER")
    def validate_directories(cls, v):
        if not isinstance(v, Path):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v


settings = Settings()  # type: ignore
