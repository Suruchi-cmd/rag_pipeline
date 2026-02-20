"""
setup_db.py — One-time database initialisation.

Creates the aerosports_rag database (if it doesn't exist), installs the
pgvector extension, and creates all tables and indexes.

Usage:
    python setup_db.py
"""

from __future__ import annotations

import logging
import sys

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL = f"""
-- Extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main knowledge store
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id           TEXT PRIMARY KEY,
    category     TEXT NOT NULL,
    subcategory  TEXT NOT NULL,
    location     TEXT NOT NULL DEFAULT 'Scarborough',
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL,
    tags         TEXT[] NOT NULL DEFAULT '{{}}',
    embedding    vector({config.EMBEDDING_DIM}),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sync state: tracks the latest sheet version we've processed
CREATE TABLE IF NOT EXISTS sync_state (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    last_version    TEXT NOT NULL DEFAULT '1.0',
    last_synced_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log of every sync operation
CREATE TABLE IF NOT EXISTS sync_history (
    id           SERIAL PRIMARY KEY,
    change_id    TEXT NOT NULL,
    chunk_id     TEXT NOT NULL,
    change_type  TEXT NOT NULL,          -- UPDATE, ADD, DELETE
    field_changed TEXT,
    synced_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_category    ON knowledge_chunks (category);
CREATE INDEX IF NOT EXISTS idx_chunks_subcategory ON knowledge_chunks (subcategory);
CREATE INDEX IF NOT EXISTS idx_chunks_tags        ON knowledge_chunks USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding   ON knowledge_chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- Seed sync_state with a single row if missing
INSERT INTO sync_state (id, last_version)
VALUES (1, '1.0')
ON CONFLICT (id) DO NOTHING;
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_exists(admin_conn, dbname: str) -> bool:
    with admin_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        return cur.fetchone() is not None


def _create_database(admin_conn, dbname: str) -> None:
    admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with admin_conn.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{dbname}"')
    logger.info("Database %r created", dbname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def setup() -> None:
    # 1. Connect to the default 'postgres' DB to check/create our target DB
    logger.info("Connecting to postgres (admin) to verify target database …")
    try:
        admin_conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname="postgres",
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
    except psycopg2.OperationalError as exc:
        logger.error("Cannot connect to PostgreSQL: %s", exc)
        sys.exit(1)

    if not _db_exists(admin_conn, config.DB_NAME):
        _create_database(admin_conn, config.DB_NAME)
    else:
        logger.info("Database %r already exists", config.DB_NAME)

    admin_conn.close()

    # 2. Connect to the target DB and run DDL
    logger.info("Applying schema to %r …", config.DB_NAME)
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(DDL)
        logger.info("Schema applied successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    setup()
    logger.info("Done. Run `python ingest.py` to load the knowledge base.")
