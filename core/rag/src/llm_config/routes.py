"""Routes for runtime chat LLM config."""

import logging

from fastapi import APIRouter, Depends, Request

from .dependencies import get_llm_config_service
from .schemas import LLMConfigResponse, LLMConfigUpdateRequest
from .services import LLMConfigService

logger = logging.getLogger(__name__)

llm_config_router = APIRouter(prefix="/settings", tags=["Settings"])


@llm_config_router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_config(
    service: LLMConfigService = Depends(get_llm_config_service),
) -> LLMConfigResponse:
    """Return current chat LLM config."""
    return LLMConfigResponse.model_validate(service.get())


@llm_config_router.put("/llm", response_model=LLMConfigResponse)
async def update_llm_config(
    body: LLMConfigUpdateRequest,
    request: Request,
    service: LLMConfigService = Depends(get_llm_config_service),
) -> LLMConfigResponse:
    """Persist new chat LLM config and hot-reload the running RAG pipeline."""
    updated = service.update(
        ollama_base_url=body.ollama_base_url,
        chat_model=body.chat_model,
        request_timeout=body.request_timeout,
    )

    rag_repo = getattr(request.app.state, "rag_repo", None)
    if rag_repo is not None:
        rag_repo.reload_chat_model(updated)
        logger.info(
            "Hot-reloaded chat model: %s @ %s",
            updated.chat_model,
            updated.ollama_base_url,
        )
    else:
        logger.warning("No rag_repo on app.state; config saved but not hot-reloaded")

    return LLMConfigResponse.model_validate(updated)
