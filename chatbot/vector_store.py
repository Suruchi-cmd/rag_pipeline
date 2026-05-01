"""Direct pgvector integration — replaces the external RAG API HTTP call."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Owns both ingestion and retrieval against pgvector.

    Call `initialize()` once at startup. All other methods are safe to call
    even if initialization failed — they return empty results / False.
    """

    def __init__(self) -> None:
        self.engine: Engine | None = None
        self._pg: PGVectorStore | None = None
        self._index: VectorStoreIndex | None = None
        self._embed_dim: int = settings.EMBED_DIM
        self._ready = False

    # ── Setup ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        try:
            self._setup_embed_model()
            self._setup_pg()
            self._ready = True
            logger.info(
                "VectorStore ready (table=%s, dim=%s)",
                settings.VECTOR_TABLE_NAME,
                self._embed_dim,
            )
        except Exception as exc:
            logger.error("VectorStore init failed — vector search disabled: %s", exc)
            self._ready = False

    def _setup_embed_model(self) -> None:
        self._embed_dim = settings.EMBED_DIM
        Settings.embed_model = OllamaEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            base_url=settings.ollama_embed_url,
        )
        logger.info(
            "Embed model: %s (dim=%s)", settings.EMBEDDING_MODEL, self._embed_dim
        )

    def _setup_pg(self) -> None:
        self.engine = create_engine(settings.pg_database_url, echo=False)
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            with suppress(Exception):
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()

        self._pg = PGVectorStore.from_params(
            database=settings.PG_DATABASE,
            host=settings.PG_HOST,
            password=settings.PG_PASSWORD,
            port=str(settings.PG_PORT),
            user=settings.PG_USER,
            table_name=settings.VECTOR_TABLE_NAME,
            embed_dim=self._embed_dim,
        )
        self._index = None

    def _ensure_index(self) -> VectorStoreIndex:
        if self._index is None:
            if not self._pg:
                raise RuntimeError("Vector store not initialized")
            self._index = VectorStoreIndex.from_vector_store(self._pg)
        return self._index

    # ── Retrieval ──────────────────────────────────────────────────────────

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Vector search. Returns list[dict] matching existing rag_docs format."""
        print(f"--" * 20)
        print(f"Incoming Query: {query}")
        print(f"--" * 20)

        if not self._ready:
            logger.warning("VectorStore not ready — returning empty results")
            return []
        try:
            return await asyncio.to_thread(self._retrieve_sync, query, top_k)
        except Exception as exc:
            logger.error("retrieve failed: %s", exc)
            return []

    def _retrieve_sync(self, query: str, top_k: int) -> list[dict]:
        index = self._ensure_index()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        results = []
        for node in nodes:
            meta = getattr(node, "metadata", {}) or {}
            results.append(
                {
                    "content": node.text,
                    "score": getattr(node, "score", 0.0),
                    "metadata": {
                        "file_name": meta.get("name", "unknown"),
                        "page": None,
                        "source": meta.get("name"),
                    },
                }
            )
        logger.info("Retrieved %d docs for query: %r", len(results), query[:60])
        return results

    # ── Ingestion ──────────────────────────────────────────────────────────

    def index_chunk(self, chunk_id: int, name: str, content: str) -> bool:
        """Embed and store a single chunk. chunk_id is used as the node id."""
        if not self._ready or not self._pg:
            return False
        try:
            clean = content.replace("\x00", "")
            if not clean.strip():
                return False
            node = TextNode(
                text=clean,
                id_=str(chunk_id),
                metadata={"name": name, "doc_id": str(chunk_id)},
            )
            storage_ctx = StorageContext.from_defaults(vector_store=self._pg)
            VectorStoreIndex(
                nodes=[node],
                storage_context=storage_ctx,
                embed_model=Settings.embed_model,
                show_progress=False,
            )
            self._index = None
            logger.info("Indexed chunk id=%s name=%r", chunk_id, name)
            return True
        except Exception as exc:
            logger.error("index_chunk failed: %s", exc)
            return False

    def delete_chunk(self, chunk_id: int) -> bool:
        """Remove a chunk from the vector store by its node id."""
        if not self._ready or not self._pg:
            return False
        try:
            self._pg.delete_nodes([str(chunk_id)])
            self._index = None
            logger.info("Deleted chunk id=%s from vector store", chunk_id)
            return True
        except AttributeError:
            # Older llama_index builds: fall back to metadata doc_id delete
            try:
                self._pg.delete(str(chunk_id))
                self._index = None
                return True
            except Exception as exc:
                logger.error("delete_chunk fallback failed: %s", exc)
                return False
        except Exception as exc:
            logger.error("delete_chunk failed: %s", exc)
            return False

    def resync(self, chunks: list) -> bool:
        """Clear all vectors then re-embed every chunk. Blocking — run in thread."""
        if not self._ready or not self.engine or not self._pg:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(f"DROP TABLE IF EXISTS data_{settings.VECTOR_TABLE_NAME}")
                )
                conn.commit()
            self._pg = None
            self._index = None
            self._setup_pg()

            if not chunks:
                logger.info("Resync: no chunks to index")
                return True

            nodes = []
            for chunk in chunks:
                clean = chunk.content.replace("\x00", "")
                if not clean.strip():
                    continue
                nodes.append(
                    TextNode(
                        text=clean,
                        id_=str(chunk.id),
                        metadata={"name": chunk.name, "doc_id": str(chunk.id)},
                    )
                )

            if nodes:
                storage_ctx = StorageContext.from_defaults(vector_store=self._pg)
                VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_ctx,
                    embed_model=Settings.embed_model,
                    show_progress=True,
                )
                self._index = None

            logger.info("Resync complete: indexed %d chunks", len(nodes))
            return True
        except Exception as exc:
            logger.error("resync failed: %s", exc)
            return False


vector_store = VectorStore()
