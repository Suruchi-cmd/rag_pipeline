from fastapi import APIRouter, Depends

from src.dependencies import get_rag_service
from src.schemas import (
    DocumentCountResponse,
    HealthStatusResponse,
    QueryRequest,
    QueryResponse,
    ResyncRequest,
    ResyncResponse,
    RetrieveResponse,
)

from .services import RAGService

rag_router = APIRouter(prefix="/rag", tags=["RAG"])


@rag_router.post("/query", response_model=QueryResponse)
async def query(
    query_request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """
    Query the RAG system with a text query.

    Args:
        query_request: The query request containing the query text and top_k
        rag_service: The RAG service dependency

    Returns:
        QueryResponse: The response containing chat response and source documents
    """
    result = rag_service.query(query_request)
    return result


@rag_router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    query_request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RetrieveResponse:
    """
    Vector search only — no LLM generation. Fast path for voice/real-time use.

    Returns ranked source documents without running the chat model.
    """
    docs = rag_service.retrieve(query_request)
    return RetrieveResponse(source_documents=docs)


@rag_router.get("/health", response_model=HealthStatusResponse)
async def get_health_status(
    include_index: bool = False,
    rag_service: RAGService = Depends(get_rag_service),
) -> HealthStatusResponse:
    """
    Get health status of the RAG system components.

    Args:
        include_index: Whether to include vector store index status in health check
        rag_service: The RAG service dependency

    Returns:
        HealthStatusResponse: Health status of vector store, embedding model, and chat model
    """
    health_status = rag_service.get_health_status(include_index=include_index)
    return HealthStatusResponse(
        vector_store=health_status.get("vector_store", False),
        embedding_model=health_status.get("embedding_model", False),
        chat_model=health_status.get("chat_model", False),
        index_status=health_status.get("index_status") if include_index else None,
    )


@rag_router.get("/documents/count", response_model=DocumentCountResponse)
async def get_document_count(
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentCountResponse:
    """Total number of documents in the vector store."""
    count = rag_service.get_document_count()
    return DocumentCountResponse(
        document_count=count, message=f"Vector store contains {count} documents"
    )


@rag_router.post("/resync", response_model=ResyncResponse)
async def resync_knowledge_base(
    body: ResyncRequest | None = None,
    rag_service: RAGService = Depends(get_rag_service),
) -> ResyncResponse:
    """Re-pull knowledge base from Google Sheets, run LLM enrichment, drop old
    embeddings, re-index from the enriched markdown.

    Blocking endpoint — parser + enrichment + embedding can take many minutes.
    """
    force = body.force_download if body else True
    result = rag_service.resync(force_download=force, enrich=True)
    return ResyncResponse(**result)


@rag_router.post("/resync-raw", response_model=ResyncResponse)
async def resync_knowledge_base_raw(
    body: ResyncRequest | None = None,
    rag_service: RAGService = Depends(get_rag_service),
) -> ResyncResponse:
    """Re-pull knowledge base from Google Sheets and re-index the RAW parser
    output, skipping the LLM enrichment step.

    Faster than `/resync` (no LLM calls) but the indexed text is the original
    table-heavy markdown rather than the voicebot-friendly prose. Useful for
    quick refreshes or when the enrichment LLM is unavailable.
    """
    force = body.force_download if body else True
    result = rag_service.resync(force_download=force, enrich=False)
    return ResyncResponse(**result)
