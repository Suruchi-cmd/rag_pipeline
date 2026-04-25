"""Dependency injection for RAG components.

RAGRepository is initialized once in lifespan() and stored on app.state.
"""

from fastapi import Request

from src.history import get_history_service
from src.rag import RAGRepository, RAGService


def get_rag_repository(request: Request) -> RAGRepository:
    rag_repo = getattr(request.app.state, "rag_repo", None)
    if rag_repo is None:
        raise RuntimeError("RAGRepository not initialized - lifespan did not run")
    return rag_repo


def get_rag_service(request: Request) -> RAGService:
    return RAGService(
        rag_repository=get_rag_repository(request),
        history_service=get_history_service(),
    )
