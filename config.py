"""
Project-wide configuration — single source of truth for the chatbot voice
server, the in-repo enrichment pipeline, and any shared utilities.

Two-service split (read this before adding settings)
====================================================
This repo runs two services as separate processes:

  1. **chatbot voice server** (this file) — runs from the repo root, loads
     the top-level `.env`, uses bare env-var names (no prefix). Everything
     defined here is for that service.

  2. **core/rag/** — ships as its own Docker image with its own `.env`
     (`core/rag/.env`) and `APP_*`-prefixed settings. Those live in
     `core/rag/src/config.py` and are NOT loaded here, because the
     container's build context is `core/rag/` and cannot reach repo-root
     files. Mirror entries are documented at the bottom of this file
     under `RAG_SETTINGS_REFERENCE` so you have one place to see every
     knob in the project.

Usage
=====
    from config import settings
    base = settings.OLLAMA_BASE_URL

The legacy import path `from chatbot.config import settings` still works
via a re-export shim in `chatbot/config.py` — both reach the same object.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Public-facing URLs ──────────────────────────────────────────────
    BASE_URL: str = Field(
        default="https://w9vm3f3crw21.share.zrok.io",
        description="Public HTTPS URL (zrok/ngrok) Twilio dials into.",
    )
    RAG_API_URL: str = Field(
        default="http://localhost:8000",
        description="In-house RAG service base URL (core/rag).",
    )

    # ── Ollama (OpenAI-compatible API surface) ──────────────────────────
    # Chatbot uses Ollama via the OpenAI SDK, which appends /chat/completions
    # under the /v1/* shim — so this URL must end with /v1.
    # The RAG/enricher side uses LlamaIndex/LangChain native Ollama clients
    # which hit /api/* directly; those URLs must NOT include /v1.
    OLLAMA_BASE_URL: str = Field(
        default="http://192.168.50.150:11434/v1",
        description="Ollama OpenAI-compatible endpoint. Must end with /v1.",
    )
    OLLAMA_KEEP_ALIVE: int = Field(
        default=-1,
        description="Seconds to keep model loaded; -1 = never unload.",
    )

    # Models. Kept split so a smaller/faster model can drive the rewriter
    # and end-of-call classifier without touching the main reply model.
    LLM_MODEL: str = Field(default="phi4:latest")
    VOICE_LLM_MODEL: str = Field(default="phi4:latest")
    VOICE_FAST_MODEL: str = Field(default="phi4:latest")

    # ── Sync chat client tuning (chatbot/llm.py) ────────────────────────
    LLM_MAX_TOKENS: int = Field(default=1024)
    LLM_TEMPERATURE: float = Field(default=0.2)
    LLM_TOP_P: float = Field(default=0.9)
    LLM_RETRIES: int = Field(default=3, description="Total attempts incl. first try.")

    # ── Voice streaming reply ───────────────────────────────────────────
    VOICE_MAX_TOKENS: int = Field(default=500)
    VOICE_TEMPERATURE: float = Field(default=0.2)
    VOICE_TOP_K: int = Field(default=7, description="RAG top_k per voice turn.")

    # ── Rewrite + end-of-call classifier (small deterministic hops) ─────
    REWRITE_MAX_TOKENS: int = Field(default=150)
    REWRITE_TEMPERATURE: float = Field(default=0.0)
    REWRITE_HISTORY_TURNS: int = Field(
        default=6,
        description="Last N messages included when rewriting follow-ups.",
    )

    # ── Conversation / context ──────────────────────────────────────────
    LLM_HISTORY_TURNS: int = Field(
        default=20,
        description="Last N messages appended to the LLM prompt per turn.",
    )
    MAX_CONVERSATION_TURNS: int = Field(
        default=30,
        description="In-memory store cap; each turn = user + assistant.",
    )
    SESSION_TIMEOUT: int = Field(
        default=30 * 60,
        description="Seconds of inactivity before a session is purged.",
    )
    SESSION_CLEANUP_INTERVAL: int = Field(
        default=5 * 60,
        description="Seconds between background cleanup sweeps.",
    )

    # ── TTS pacing — 'fast first, smooth after' ─────────────────────────
    TTS_FIRST_FLUSH_WORDS: int = Field(default=8)
    TTS_FIRST_FLUSH_CHARS: int = Field(default=60)

    # ── HTTP ────────────────────────────────────────────────────────────
    RAG_HTTP_TIMEOUT: float = Field(default=15.0)

    # ── Locale ──────────────────────────────────────────────────────────
    TIMEZONE: str = Field(
        default="America/Toronto",
        description="ZoneInfo name; injected into the system prompt as 'current time'.",
    )

    # ── CORS ────────────────────────────────────────────────────────────
    CHATBOT_CORS_ORIGINS: str = Field(
        default="*",
        description="Comma-separated origin list. '*' allows all (dev only).",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CHATBOT_CORS_ORIGINS.split(",") if o.strip()]

    # ── Twilio ConversationRelay TwiML attributes ───────────────────────
    TWILIO_LANGUAGE: str = Field(default="en-CA")
    TWILIO_ASR_PROVIDER: str = Field(default="deepgram")
    TWILIO_TTS_PROVIDER: str = Field(default="ElevenLabs")
    TWILIO_VOICE_ID: str = Field(default="uYXf8XasLslADfZ2MB4u")
    TWILIO_INTERRUPT_SENSITIVITY: str = Field(default="medium")

    # Long ASR-hints string. Pulled out so the operations team can update
    # park-specific terms without redeploying. Single-line; commas separate.
    TWILIO_ASR_HINTS: str = Field(
        default=(
            "AeroSports, Aerosports, Aero Sports, Scarborough, Birchmount, "
            "aerosportsparks, Aero Slam, birthday, Aero Camp, Ninja Maze, "
            "Ninja Warrior, 360 Bicycle, Donut Slide, Carpet Slide, go kart, "
            "go karting, dual kart, Mini Track, Main Track, Premium Jump Pass, "
            "VIP Jump Pass, All Day Pass, 30 Day Pass, 90 Day Pass, Annual Pass, "
            "Premium Party, VIP Party, Ultimate Party, Toddler Time, PA Day Camp, "
            "March Break Camp, availability, Special Needs Program, "
            "Birds Eye Party Arena, jump tower, slam basketball, dodgeball, "
            "rock walls, waiver"
        ),
    )

    # ── Twilio account creds (used by Twilio SDK clients) ───────────────
    TWILIO_ACCOUNT_SID: str = Field(default="")
    TWILIO_AUTH_TOKEN: str = Field(default="")
    TWILIO_PHONE_NUMBER: str = Field(default="")

    # ── Business facts (used in greetings + fallback strings) ───────────
    BUSINESS_NAME: str = Field(default="AeroSports Scarborough")
    AGENT_NAME: str = Field(default="Maya")

    # Literal forms — used when the LLM consumes them as instructions.
    BUSINESS_PHONE: str = Field(default="289-454-5555")
    BUSINESS_EMAIL: str = Field(default="events.scb@aerosportsparks.ca")

    # Spoken (TTS-friendly) forms — used in canned strings sent to the caller.
    BUSINESS_PHONE_SPOKEN: str = Field(
        default="two eight nine, four five four, five five five five"
    )
    BUSINESS_EMAIL_SPOKEN: str = Field(
        default="events dot scb at aerosportsparks dot c a"
    )

    # Echo-detection prefix used to suppress ASR catching the bot's own
    # welcome greeting. Defaults aligned with the default greeting; override
    # if you customise BUSINESS_NAME / AGENT_NAME.
    WELCOME_ECHO_PREFIX: str = Field(default="thank you for calling aerosports")

    @property
    def welcome_greeting(self) -> str:
        return (
            f"Thank you for calling {self.BUSINESS_NAME}, this is "
            f"{self.AGENT_NAME}. How can I help you?"
        )

    @property
    def fallback_message(self) -> str:
        return (
            "I'm so sorry, I'm having a little trouble on my end right now. "
            "Can you try calling back in just a minute? Or you can reach us "
            f"at {self.BUSINESS_PHONE_SPOKEN}. Sorry about that!"
        )

    # ── Logging ────────────────────────────────────────────────────────
    SESSION_LOG_DIR: str = Field(default="logs")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # type: ignore[call-arg]


# ════════════════════════════════════════════════════════════════════════
# RAG_SETTINGS_REFERENCE
# ════════════════════════════════════════════════════════════════════════
# The settings below are loaded by core/rag/src/config.py inside the RAG
# Docker container. They are documented here ONLY for visibility — this
# module does not read them. To change one, edit core/rag/.env.
#
# All keys carry the APP_ prefix in the environment.
#
# Postgres / pgvector
# -------------------
#   APP_PG_HOST              host of the pgvector Postgres
#   APP_PG_PORT              port (5433 in the bundled docker-compose)
#   APP_PG_USER              postgres user
#   APP_PG_PASSWORD          postgres password
#   APP_PG_DATABASE          postgres database
#   APP_VECTOR_TABLE_NAME    pgvector table name (default: documents)
#
# Embedding model (static — changing requires re-index)
# ------------------------------------------------------
#   APP_EMBEDDING_MODEL      default: embeddinggemma
#   APP_EMBED_DIM            embedding dimension; default: 768
#
# LLM bootstrap (seeds the LLMConfig DB row on first boot only)
# --------------------------------------------------------------
#   APP_DEFAULT_OLLAMA_BASE_URL   native Ollama URL — NO /v1
#   APP_DEFAULT_CHAT_MODEL        default: gemma3:12b
#   APP_DEFAULT_REQUEST_TIMEOUT   seconds; default: 120
#
# Indexing / chunking
# -------------------
#   APP_LLM_TEMPERATURE      passed to Ollama for chat completions
#   APP_CHUNK_SIZE           SentenceSplitter chunk size; default: 1000
#   APP_CHUNK_OVERLAP        SentenceSplitter overlap; default: 200
#
# Retrieval
# ---------
#   APP_RETRIEVAL_OVERSAMPLE multiplier on top_k before trimming; default: 2
#   APP_RETRIEVAL_MAX_K      hard cap after oversampling; default: 15
#   APP_SIMILARITY_CUTOFF    rerank threshold; default: 0.6
#   APP_RESPONSE_MODE        LlamaIndex synthesizer mode; default: compact
#
# Resync subprocess timeouts
# --------------------------
#   APP_PARSER_TIMEOUT_ENRICH    seconds; default: 1800
#   APP_PARSER_TIMEOUT_RAW       seconds; default: 300
#
# Bootstrap behaviour
# -------------------
#   APP_AUTO_INGEST          re-ingest enriched markdown on cold boot
#   APP_DB_WAIT_ATTEMPTS     poll attempts during pgvector cold start
#   APP_DB_WAIT_DELAY        seconds between polls
#
# Server bindings
# ---------------
#   APP_HOST, APP_PORT, APP_RELOAD, APP_WORKERS, APP_LOG_LEVEL
#
# CORS
# ----
#   APP_CORS_ORIGINS         comma-separated list
#
# ════════════════════════════════════════════════════════════════════════
# PARSER_SETTINGS_REFERENCE
# ════════════════════════════════════════════════════════════════════════
# The parser/enricher pipeline (core/rag/parser/) is invoked as a
# subprocess by core/rag's /rag/resync endpoint. It currently uses CLI
# flags + hardcoded constants rather than env-driven settings, so the
# values below are NOT read by this module — they are listed here only
# as a single-page map of every knob in the parser path.
#
# To change one, either pass the matching CLI flag (--force, --model,
# --base-url, etc.) or edit the hardcoded constant in the source file.
#
#   Spreadsheet ID            run_parsers.py:37  (constant SPREADSHEET_ID)
#   Output dir                run_parsers.py:39  (constant OUTPUT_DIR)
#   Enrichment model          run_parsers.py     CLI: --model
#   Enrichment Ollama URL     run_parsers.py     CLI: --base-url   (no /v1)
#   GCP service-account JSON  credentials/google_service_account.json
#                             (path is hardcoded in sheet_downloader logic)
#   SQLite call log path      db.py              parameter db_path="calls.db"
#
# Wiring these into env-driven settings requires either (a) reading the
# repo-root .env from inside core/rag's Docker container (needs Dockerfile
# build-context change), or (b) loading a separate parser-side .env.
# Tracked as a TODO; left out here to avoid defining settings that no
# code actually reads.
# ════════════════════════════════════════════════════════════════════════
