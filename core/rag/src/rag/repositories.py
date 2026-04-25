"""Repositories module for RAG database operations.

Vector storage, document indexing, and query operations. The chat LLM
settings come from an LLMConfig row (runtime-mutable); embedding model
and DB settings are static from src.config.settings.
"""

import logging
import traceback
from contextlib import suppress

from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import settings
from src.llm_config.models import LLMConfig
from src.schemas import DocumentMetadata, QueryRequest, QueryResponse, SourceDocument

logger = logging.getLogger(__name__)


class RAGRepository:
    """Repository class for RAG database operations."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self.engine: None | Engine = None
        self.vector_store: None | PGVectorStore = None
        self.storage_context: None | StorageContext = None
        self.index: None | VectorStoreIndex = None
        self._actual_embed_dim: int | None = None
        self._llm_config = llm_config
        self._setup_models()
        self._setup_database()

    def _setup_models(self) -> None:
        """Setup the LLM and embedding models and validate embedding dimension."""
        try:
            Settings.llm = Ollama(
                model=self._llm_config.chat_model,
                base_url=self._llm_config.ollama_base_url,
                request_timeout=float(self._llm_config.request_timeout),
                temperature=settings.LLM_TEMPERATURE,
            )
            Settings.embed_model = OllamaEmbedding(
                model_name=settings.EMBEDDING_MODEL,
                base_url=self._llm_config.ollama_base_url,
            )
            logger.info(
                "Models configured: LLM=%s, Embedding=%s, Ollama=%s",
                self._llm_config.chat_model,
                settings.EMBEDDING_MODEL,
                self._llm_config.ollama_base_url,
            )

            try:
                test_vector = Settings.embed_model.get_text_embedding("__dim_probe__")
                self._actual_embed_dim = len(test_vector)
                if self._actual_embed_dim != settings.EMBED_DIM:
                    logger.warning(
                        "Configured EMBED_DIM (%s) does not match model output dimension (%s). Using model output dimension.",
                        settings.EMBED_DIM,
                        self._actual_embed_dim,
                    )
                else:
                    logger.info(
                        "Embedding dimension confirmed: %s", self._actual_embed_dim
                    )
            except Exception as e:
                logger.error("Failed to probe embedding dimension: %s", e)
                self._actual_embed_dim = None

        except Exception as e:
            logger.error("Failed to setup models: %s", e)
            raise

    def reload_chat_model(self, llm_config: LLMConfig) -> None:
        """Rebuild Settings.llm + embedding base_url from new config.

        Embedding model itself stays static - only base_url follows the
        ollama host so a single Ollama deployment handles both.
        """
        self._llm_config = llm_config
        Settings.llm = Ollama(
            model=llm_config.chat_model,
            base_url=llm_config.ollama_base_url,
            request_timeout=float(llm_config.request_timeout),
            temperature=settings.LLM_TEMPERATURE,
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            base_url=llm_config.ollama_base_url,
        )
        logger.info(
            "Chat model reloaded: %s @ %s (timeout=%ss)",
            llm_config.chat_model,
            llm_config.ollama_base_url,
            llm_config.request_timeout,
        )

    def _setup_database(self) -> None:
        """Setup the database connection, extension and vector store."""
        try:
            self.engine = create_engine(settings.database_url, echo=False)
            if self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    logger.info("Database connection established")
                    with suppress(Exception):
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                        conn.commit()
                        logger.info("pgvector extension ensured")

            embed_dim = self._actual_embed_dim or settings.EMBED_DIM
            if self._actual_embed_dim is None:
                logger.warning(
                    "Using configured EMBED_DIM=%s (model dimension probe failed earlier)",
                    settings.EMBED_DIM,
                )

            self.vector_store = PGVectorStore.from_params(
                database=settings.PG_DATABASE,
                host=settings.PG_HOST,
                password=settings.PG_PASSWORD,
                port=str(settings.PG_PORT),
                user=settings.PG_USER,
                table_name=settings.VECTOR_TABLE_NAME,
                embed_dim=embed_dim,
            )
            logger.info(
                "Vector store configured with table '%s' (embed_dim=%s)",
                settings.VECTOR_TABLE_NAME,
                embed_dim,
            )
        except Exception as e:
            logger.error("Failed to setup database: %s", e)
            raise

    def index_documents(self, documents: list) -> bool:
        """Index documents into the vector store.

        Uses MarkdownNodeParser to respect document structure (headers),
        then SentenceSplitter for further chunking of large sections.
        """
        try:
            logger.info("Creating index from documents...")
            logger.info(f"Number of documents to index: {len(documents)}")

            sanitized_documents = []
            for doc in documents:
                if hasattr(doc, "text") and doc.text:
                    sanitized_text = doc.text.replace("\x00", "")
                    sanitized_doc = Document(
                        text=sanitized_text,
                        metadata=doc.metadata if hasattr(doc, "metadata") else {},
                        id_=doc.id_ if hasattr(doc, "id_") else None,
                    )
                    sanitized_documents.append(sanitized_doc)
                else:
                    sanitized_documents.append(doc)

            logger.info(f"Sanitized {len(sanitized_documents)} documents")

            markdown_parser = MarkdownNodeParser(
                include_metadata=True, include_prev_next_rel=True
            )

            text_splitter = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separator=" ",
            )

            logger.info("Parsing documents with MarkdownNodeParser...")
            initial_nodes = markdown_parser.get_nodes_from_documents(
                sanitized_documents
            )
            logger.info(
                f"Created {len(initial_nodes)} initial nodes from markdown structure"
            )

            final_nodes = []
            for node in initial_nodes:
                if len(node.text) > settings.CHUNK_SIZE:
                    temp_doc = Document(
                        text=node.text, metadata=node.metadata, id_=node.id_
                    )
                    sub_nodes = text_splitter.get_nodes_from_documents([temp_doc])
                    for sub_node in sub_nodes:
                        sub_node.metadata.update(node.metadata)
                    final_nodes.extend(sub_nodes)
                else:
                    final_nodes.append(node)

            logger.info(f"Final node count after chunking: {len(final_nodes)}")

            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            self.index = VectorStoreIndex(
                nodes=final_nodes,
                storage_context=self.storage_context,
                embed_model=Settings.embed_model,
                show_progress=True,
            )

            if self.index:
                try:
                    docstore = self.index.docstore
                    node_count = (
                        len(docstore.docs) if hasattr(docstore, "docs") else "unknown"
                    )
                    logger.info(
                        f"Documents indexed successfully - Created {node_count} text chunks"
                    )
                except Exception as e:
                    logger.info("Documents indexed successfully")
                    logger.debug(f"Could not retrieve node count: {e}")

            return True

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            traceback.print_exc()
            return False

    def query(self, query_request: QueryRequest) -> QueryResponse:
        """Query the RAG system."""
        try:
            health = self.health_check(require_index=False)
            basic_health = {k: v for k, v in health.items() if k != "index"}
            if not all(basic_health.values()):
                logger.error("System not ready for queries - basic components failed")
                logger.error(f"Health status: {health}")
                raise ValueError("System not healthy for queries")

            doc_count = self.get_document_count()
            if doc_count == 0:
                logger.error("No documents in vector store - cannot perform queries")
                raise ValueError("No documents in vector store")

            logger.info(f"Vector store contains {doc_count} documents")

            if not self.index:
                logger.info("Index not initialized, creating from vector store...")
                if self.vector_store:
                    self.index = VectorStoreIndex.from_vector_store(self.vector_store)
                    logger.info("✓ Index successfully created from vector store")
                else:
                    logger.error("Vector store not initialized")
                    raise ValueError("Vector store not initialized")

            optimized_top_k = min(
                query_request.top_k * settings.RETRIEVAL_OVERSAMPLE,
                settings.RETRIEVAL_MAX_K,
            )

            # customer_service_prompt_tmpl = (
            #     "Context information is below.\n"
            #     "---------------------\n"
            #     "{context_str}\n"
            #     "---------------------\n"
            #     "As a professional customer service representative, please provide a helpful, "
            #     "friendly, and short answer to the customer's question based on the context information provided. Explain only if required.\n\n"
            #     "Customer's Question: {query_str}\n\n"
            #     "Your Response: "
            # )
            customer_service_prompt_tmpl = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Answer the question using ONLY the information provided in the context.\n"
            "Do NOT use external knowledge, assumptions, or fill in missing details.\n"
            "If the answer is not explicitly present in the context, reply with: 'Sorry I dont know the answer.'\n"
            "Keep the response concise and directly relevant to the question.\n\n"
            "Question: {query_str}\n\n"
            "Answer: "
        )

            customer_service_prompt = PromptTemplate(customer_service_prompt_tmpl)

            # TODO : Check all the response_mode
            response_synthesizer = get_response_synthesizer(
                response_mode=settings.RESPONSE_MODE,
                # summary_template=customer_service_prompt,
                verbose=False,
            )

            query_engine = self.index.as_query_engine(
                similarity_top_k=optimized_top_k,
                response_synthesizer=response_synthesizer,
                similarity_cutoff=settings.SIMILARITY_CUTOFF,
            )

            response = query_engine.query(query_request.query)

            source_documents: list[SourceDocument] = []
            if hasattr(response, "source_nodes") and response.source_nodes:
                top_nodes = response.source_nodes[: query_request.top_k]
                for node in top_nodes:
                    node_metadata = node.metadata if hasattr(node, "metadata") else {}
                    file_name = (
                        node_metadata.get("file_name")
                        or node_metadata.get("filename")
                        or node_metadata.get("file_path", "").split("/")[-1]
                        or "Unknown Document"
                    )
                    page = (
                        node_metadata.get("page")
                        or node_metadata.get("page_number")
                        or node_metadata.get("page_label")
                    )

                    if isinstance(page, str) and page.isdigit():
                        page = int(page)
                    elif not isinstance(page, int):
                        page = None

                    structured_metadata = DocumentMetadata(
                        file_name=file_name,
                        page=page,
                        source=node_metadata.get("file_path"),
                    )

                    source_doc = SourceDocument(
                        content=node.text,
                        score=getattr(node, "score", 0.0),
                        metadata=structured_metadata,
                    )
                    source_documents.append(source_doc)

            logger.info("Query executed successfully")
            return QueryResponse(
                chat_response=str(response), source_documents=source_documents
            )

        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def retrieve(self, query_request: QueryRequest) -> list:
        """Vector search only — no LLM generation."""
        try:
            if not self.index:
                if self.vector_store:
                    self.index = VectorStoreIndex.from_vector_store(self.vector_store)
                else:
                    raise ValueError("Vector store not initialized")

            retriever = self.index.as_retriever(
                similarity_top_k=query_request.top_k,
            )
            nodes = retriever.retrieve(query_request.query)

            source_documents = []
            for node in nodes:
                node_metadata = node.metadata if hasattr(node, "metadata") else {}
                file_name = (
                    node_metadata.get("file_name")
                    or node_metadata.get("file_path", "").split("/")[-1]
                    or "Unknown Document"
                )
                page = node_metadata.get("page") or node_metadata.get("page_number")
                if isinstance(page, str) and page.isdigit():
                    page = int(page)
                elif not isinstance(page, int):
                    page = None

                source_documents.append(
                    SourceDocument(
                        content=node.text,
                        score=getattr(node, "score", 0.0),
                        metadata=DocumentMetadata(
                            file_name=file_name,
                            page=page,
                            source=node_metadata.get("file_path"),
                        ),
                    )
                )

            logger.info(
                "Retrieve executed: %d docs for '%s'",
                len(source_documents),
                query_request.query[:50],
            )
            return source_documents

        except Exception as e:
            logger.error("Retrieve failed: %s", e)
            raise

    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store."""
        if not self.engine:
            logger.warning("get_document_count called before engine initialization")
            return 0
        try:
            with self.engine.connect() as conn:
                exists_result = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables WHERE table_name = :tbl"
                    ),
                    {"tbl": f"data_{settings.VECTOR_TABLE_NAME}"},
                ).fetchone()
                if not exists_result:
                    logger.info(
                        "Vector table '%s' not found (no rows to count yet)",
                        settings.VECTOR_TABLE_NAME,
                    )
                    return 0
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM data_{settings.VECTOR_TABLE_NAME}")
                )
                row = result.fetchone()
                count = int(row[0]) if row and row[0] is not None else 0
                return count
        except Exception as e:
            logger.error(
                "Failed to get document count from table '%s': %s",
                settings.VECTOR_TABLE_NAME,
                e,
            )
            return 0

    def clear_index(self) -> bool:
        """Clear all documents from the index."""
        try:
            if not self.vector_store or not self.engine:
                return False

            with self.engine.connect() as conn:
                conn.execute(
                    text(f"DROP TABLE IF EXISTS data_{settings.VECTOR_TABLE_NAME}")
                )
                conn.commit()

            self._setup_database()
            self.index = None

            logger.info("Index cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False

    def force_recreate_index(self) -> bool:
        """Force drop and recreate the vector table (dimension mismatch recovery)."""
        if not self.engine:
            logger.error("Cannot recreate index: engine not initialized")
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(f"DROP TABLE IF EXISTS data_{settings.VECTOR_TABLE_NAME}")
                )
                conn.commit()
            logger.info(
                "Dropped existing vector table '%s'", settings.VECTOR_TABLE_NAME
            )
            self.vector_store = None
            self.index = None
            self._setup_database()
            return True
        except Exception as e:
            logger.error("Failed to force recreate index: %s", e)
            return False

    def health_check(self, require_index: bool = False) -> dict:
        """Perform a health check on the repository."""
        health = {
            "database": False,
            "vector_store": False,
            "models": False,
            "index": False,
        }

        try:
            if self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                health["database"] = True

            health["vector_store"] = self.vector_store is not None

            health["models"] = (
                Settings.llm is not None and Settings.embed_model is not None
            )

            if require_index:
                health["index"] = self.index is not None
            else:
                health["index"] = True

        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return health
