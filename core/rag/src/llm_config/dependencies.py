from .repositories import LLMConfigRepository
from .services import LLMConfigService


def get_llm_config_repository() -> LLMConfigRepository:
    return LLMConfigRepository()


def get_llm_config_service() -> LLMConfigService:
    return LLMConfigService(repository=get_llm_config_repository())
