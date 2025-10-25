"""Embeddings initialization and management."""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ..models.config import EmbeddingConfig, EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Manages embeddings initialization."""

    def __init__(self, embedding_config: EmbeddingConfig):
        """Initialize embeddings manager.

        Args:
            embedding_config: Embedding configuration.
        """
        self.embedding_config = embedding_config
        self._embeddings: Optional[Embeddings] = None

    def get_embeddings(self) -> Embeddings:
        """Get or create embeddings instance.

        Returns:
            Embeddings instance.

        Raises:
            ValueError: If provider is not supported.
        """
        if self._embeddings is None:
            logger.info(
                f"Initializing {self.embedding_config.provider} embeddings "
                f"with model: {self.embedding_config.model_name}"
            )

            if self.embedding_config.provider == EmbeddingProvider.OPENAI:
                self._embeddings = self._create_openai_embeddings()
            else:
                raise ValueError(
                    f"Unsupported embedding provider: {self.embedding_config.provider}"
                )

            logger.info("Embeddings initialized successfully")

        return self._embeddings

    def _create_openai_embeddings(self) -> OpenAIEmbeddings:
        """Create OpenAI embeddings instance.

        Returns:
            OpenAIEmbeddings instance.
        """
        kwargs = {
            "model": self.embedding_config.model_name,
        }

        if self.embedding_config.api_key:
            kwargs["openai_api_key"] = self.embedding_config.api_key

        # Handle dimensions parameter for models that support it
        if self.embedding_config.dimensions is not None:
            # Only certain models support custom dimensions
            if "text-embedding-3" in self.embedding_config.model_name:
                kwargs["dimensions"] = self.embedding_config.dimensions

        return OpenAIEmbeddings(**kwargs)
