"""
Service layer for handling business logic related to RAG operations.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
)

from src.config import BASE_DIR, settings
from src.history import HistoryService
from src.schemas import QueryRequest, QueryResponse

from .repositories import RAGRepository

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, rag_repository: RAGRepository, history_service: HistoryService):
        self.rag_repository = rag_repository
        self.history_service = history_service

    def get_health_status(self, include_index: bool = False) -> dict:
        """Get the health status of the RAG repository.

        Returns:
            dict: Health status: 'vector_store', 'embedding_model', 'chat_model'
        """
        return self.rag_repository.health_check(require_index=include_index)

    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store.

        Returns:
            int: Total document count
        """
        return self.rag_repository.get_document_count()

    def index_documents(self, documents: list) -> bool:
        """Index a list of documents.

        Args:
            documents: List of Document objects to index

        Returns:
            bool: True if indexing was successful
        """
        try:
            if not documents:
                logger.warning("No documents to index")
                return False
            logger.info(f"Indexing {len(documents)} documents")
            return self.rag_repository.index_documents(documents)
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return False

    def index_documents_from_directory(
        self,
        directory_path: Path,
        recursive: bool = True,
        exclude_subdirs: tuple[str, ...] = (),
        exclude_files: tuple[str, ...] = (),
    ) -> int:
        """Index all markdown documents from a directory.

        Args:
            directory_path: Folder to scan.
            recursive: If False, only top-level *.md (used for raw ingest
                so we don't sweep `enriched/` and `original/` subdirs).
            exclude_subdirs: Subdir names whose files are skipped (recursive only).
            exclude_files: Filenames to skip outright (e.g. data_understanding.md).

        Returns:
            int: Number of documents loaded (0 on failure).
        """
        try:
            directory_path = Path(directory_path)
            if recursive:
                md_files = list(directory_path.rglob("*.md"))
                if exclude_subdirs:
                    md_files = [
                        f
                        for f in md_files
                        if not any(part in exclude_subdirs for part in f.parts)
                    ]
            else:
                md_files = list(directory_path.glob("*.md"))

            if exclude_files:
                md_files = [f for f in md_files if f.name not in exclude_files]

            if not md_files:
                logger.warning("No markdown files in %s", directory_path)
                return 0
            documents = SimpleDirectoryReader(
                input_files=[str(f) for f in md_files]
            ).load_data()
            ok = self.rag_repository.index_documents(documents)
            return len(documents) if ok else 0
        except Exception as e:
            logger.error(
                f"Failed to index documents from directory {directory_path}: {e}"
            )
            return 0

    def resync(self, force_download: bool = True, enrich: bool = True) -> dict:
        """Re-pull the knowledge base and rebuild the vector store.

        Order of operations (wipe-first):
          1. Drop + recreate the pgvector table — clears every existing row
             and forces a clean schema for the new ingest.
          2. Subprocess the parser. With `enrich=True` it also runs the
             two-phase LLM enrichment pipeline into `parser/output/enriched/`.
             With `enrich=False` it only downloads and writes the raw
             markdown into `parser/output/`.
          3. Re-index from the appropriate folder:
               - enrich=True  → settings.DATA_FOLDER       (parser/output/enriched)
               - enrich=False → settings.RAW_DATA_FOLDER   (parser/output, top-level)

        NOTE: Wiping first means a parser failure leaves the vector store
        empty until the next successful resync. This is intentional — caller
        wants a guaranteed-clean target rather than a stale fallback.

        Parser runs in a subprocess because parser/src/ shadows the backend
        src/ package when imported in-process. Timeout is long because the
        LLM enrichment step can easily exceed a few minutes.
        """
        # ── Step 1: wipe the vector store FIRST ──────────────────────────
        logger.info("Resync: dropping existing vector table (wipe-first)")
        if not self.rag_repository.force_recreate_index():
            return {
                "parser_ok": False,
                "documents_indexed": 0,
                "document_count": self.rag_repository.get_document_count(),
                "message": "Vector table wipe/recreate failed — aborting before parser run",
            }
        logger.info(
            "Resync: vector table cleared (rows=%d)",
            self.rag_repository.get_document_count(),
        )

        parser_dir = BASE_DIR / "parser"
        cmd = [sys.executable, "run_parsers.py"]
        if enrich:
            cmd.append("--enrich")
        if force_download:
            cmd.append("--force")

        # Enrichment is LLM-bound and scales with workbook count × model speed.
        # 30 minutes gives headroom over current ~14-workbook runs. Raw-only
        # parsing is fast (seconds) but we keep the same ceiling for simplicity.
        PARSER_TIMEOUT = 1800 if enrich else 300

        logger.info(
            "Resync: running parser%s (cwd=%s, force=%s, timeout=%ss)",
            "+enrich" if enrich else " (raw-only)",
            parser_dir,
            force_download,
            PARSER_TIMEOUT,
        )

        # Stream stdout/stderr live into our logger so every parser + enricher
        # line shows up in the server log as it happens, instead of being
        # buffered until the subprocess exits (and then truncated to a tail).
        # PYTHONUNBUFFERED=1 forces child stdout/stderr to flush each line
        # immediately, so our line-pump thread can log as events happen
        # rather than waiting for the child to fill its buffer.
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        proc = subprocess.Popen(
            cmd,
            cwd=parser_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge so ordering is preserved
            text=True,
            bufsize=1,
            env=child_env,
        )

        collected: list[str] = []

        def _pump(pipe) -> None:
            for raw in iter(pipe.readline, ""):
                line = raw.rstrip()
                if not line:
                    continue
                collected.append(line)
                logger.info("[parser] %s", line)
            pipe.close()

        pump_thread = threading.Thread(
            target=_pump, args=(proc.stdout,), daemon=True
        )
        pump_thread.start()

        try:
            rc = proc.wait(timeout=PARSER_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pump_thread.join(timeout=5)
            logger.error("Parser timed out after %ss", PARSER_TIMEOUT)
            return {
                "parser_ok": False,
                "documents_indexed": 0,
                "document_count": self.rag_repository.get_document_count(),
                "message": f"Parser timed out after {PARSER_TIMEOUT}s",
            }

        pump_thread.join(timeout=5)
        parser_ok = rc == 0
        if not parser_ok:
            # Dump the full captured output so the operator can see the
            # full failure context (including LLM 502 HTML bodies, etc.)
            # without needing to re-run with higher verbosity.
            tail = "\n".join(collected[-200:])
            logger.error(
                "Parser failed (rc=%s). Last %d lines:\n%s",
                rc,
                min(len(collected), 200),
                tail,
            )
            return {
                "parser_ok": False,
                "documents_indexed": 0,
                "document_count": self.rag_repository.get_document_count(),
                "message": f"Parser failed (rc={rc})",
            }

        if enrich:
            target_folder = Path(settings.DATA_FOLDER)
            logger.info(
                "Resync: re-indexing enriched markdown from %s", target_folder
            )
            indexed = self.index_documents_from_directory(target_folder)
        else:
            target_folder = Path(settings.RAW_DATA_FOLDER)
            logger.info(
                "Resync: re-indexing raw markdown (top-level only) from %s",
                target_folder,
            )
            # Top-level only so we skip `enriched/` and `original/` subdirs,
            # and exclude the consolidated understanding doc.
            indexed = self.index_documents_from_directory(
                target_folder,
                recursive=False,
                exclude_files=("data_understanding.md",),
            )

        count = self.rag_repository.get_document_count()
        mode = "enriched" if enrich else "raw"
        return {
            "parser_ok": True,
            "documents_indexed": indexed,
            "document_count": count,
            "message": f"Resync ({mode}) complete: {indexed} docs loaded, {count} vector rows",
        }

    def retrieve(self, query_request: QueryRequest) -> list:
        """Vector search only — no LLM. Fast path for voice/real-time callers."""
        return self.rag_repository.retrieve(query_request)

    def query(self, query_request: QueryRequest) -> QueryResponse:
        """Query the vector store and get a response from the chat model.

        Args:
            query_request: The query request object containing query details

        Returns:
            QueryResponse: The response object containing query results
        """
        start_time = time.time()
        error_message = None
        success = True
        result = QueryResponse(
            chat_response="Error processing query.", source_documents=[]
        )
        try:
            result = self.rag_repository.query(query_request)
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Query failed for '{query_request}': {e}")
        finally:
            response_time_ms = int((time.time() - start_time) * 1000)
            try:
                self.history_service.save_query_history(
                    query_request=query_request,
                    query_response=result,
                    response_time_ms=response_time_ms,
                    success=success,
                    error_message=error_message,
                )

            except Exception as e:
                logger.error(f"Failed to save query history: {e}")
        return result
