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

    # Chat LLM tuning (passed to Ollama). Embedding model temperature is N/A.
    LLM_TEMPERATURE: float = Field(default=0.3)

    # Document chunking — used by SentenceSplitter when MarkdownNodeParser
    # produces a node larger than CHUNK_SIZE.
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    # Retrieval. We oversample (top_k * RETRIEVAL_OVERSAMPLE, capped at
    # RETRIEVAL_MAX_K) and then trim to the requested top_k after rerank.
    RETRIEVAL_OVERSAMPLE: int = Field(default=2)
    RETRIEVAL_MAX_K: int = Field(default=15)
    SIMILARITY_CUTOFF: float = Field(default=0.6)
    RESPONSE_MODE: str = Field(default="compact")

    # Resync subprocess (parser + optional LLM enrichment) timeouts in seconds.
    PARSER_TIMEOUT_ENRICH: int = Field(default=1800)
    PARSER_TIMEOUT_RAW: int = Field(default=300)

    # Bootstrap DB wait — covers pgvector container cold start.
    DB_WAIT_ATTEMPTS: int = Field(default=30)
    DB_WAIT_DELAY: float = Field(default=2.0)

    # CORS — comma-separated origin list; "*" allows all (dev only).
    CORS_ORIGINS: str = Field(
        default=(
            "http://localhost:3000,http://localhost:8023,"
            "http://127.0.0.1:3000,http://127.0.0.1:8023"
        )
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

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
