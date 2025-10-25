"""LLM initialization and management."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..models.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM initialization."""

    def __init__(self, llm_config: LLMConfig):
        """Initialize LLM manager.

        Args:
            llm_config: LLM configuration.
        """
        self.llm_config = llm_config
        self._llm: BaseChatModel | None = None

    def get_llm(self) -> BaseChatModel:
        """Get or create LLM instance.

        Returns:
            LLM instance.

        Raises:
            ValueError: If provider is not supported.
        """
        if self._llm is None:
            logger.info(
                f"Initializing {self.llm_config.provider} LLM "
                f"with model: {self.llm_config.model_name}"
            )

            if self.llm_config.provider == LLMProvider.OPENAI:
                self._llm = self._create_openai_llm()
            else:
                raise ValueError(
                    f"Unsupported LLM provider: {self.llm_config.provider}"
                )

            logger.info("LLM initialized successfully")

        return self._llm

    def _create_openai_llm(self) -> ChatOpenAI:
        """Create OpenAI LLM instance.

        Returns:
            ChatOpenAI instance.
        """
        kwargs = {
            "model": self.llm_config.model_name,
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens,
        }

        if self.llm_config.api_key:
            kwargs["openai_api_key"] = self.llm_config.api_key

        return ChatOpenAI(**kwargs)
