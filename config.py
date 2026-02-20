"""Central configuration: DB connection pool, Google Sheets client, embedding settings."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding config
# ---------------------------------------------------------------------------

EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "voyage")

# Voyage AI: 1024-dim;  sentence-transformers: 384-dim
EMBEDDING_DIM: int = 1024 if EMBEDDING_PROVIDER == "voyage" else 384

VOYAGE_API_KEY: Optional[str] = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL: str = "voyage-2"

LOCAL_MODEL_NAME: str = "all-MiniLM-L6-v2"

EMBED_BATCH_SIZE: int = 128

# ---------------------------------------------------------------------------
# Database config
# ---------------------------------------------------------------------------

DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "aerosports_rag")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

_db_pool: Optional[pool.ThreadedConnectionPool] = None


def get_db_pool() -> pool.ThreadedConnectionPool:
    """Return the singleton threaded connection pool, creating it on first call."""
    global _db_pool
    if _db_pool is None:
        _db_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        logger.info("DB pool created (host=%s db=%s)", DB_HOST, DB_NAME)
    return _db_pool


def close_db_pool() -> None:
    global _db_pool
    if _db_pool is not None:
        _db_pool.closeall()
        _db_pool = None
        logger.info("DB pool closed")


# ---------------------------------------------------------------------------
# Google Sheets config
# ---------------------------------------------------------------------------

GOOGLE_SHEET_ID: str = os.getenv("GOOGLE_SHEET_ID", "")
GOOGLE_CREDENTIALS_PATH: str = os.getenv(
    "GOOGLE_CREDENTIALS_PATH", "credentials/google_service_account.json"
)
CHANGE_LOG_SHEET: str = os.getenv("CHANGE_LOG_SHEET", "Change Log")
VERSION_CELL: str = os.getenv("VERSION_CELL", "M1")


@lru_cache(maxsize=1)
def get_sheets_client():
    """Return a cached gspread client authorised via service account."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=scopes)
    client = gspread.authorize(creds)
    logger.info("Google Sheets client initialised")
    return client


def get_spreadsheet():
    """Open and return the AeroSports spreadsheet."""
    return get_sheets_client().open_by_key(GOOGLE_SHEET_ID)


# ---------------------------------------------------------------------------
# Search config
# ---------------------------------------------------------------------------

DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
HYBRID_SEMANTIC_WEIGHT: float = 0.7
HYBRID_KEYWORD_WEIGHT: float = 0.3
