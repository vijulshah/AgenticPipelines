"""Tests for lazy import functionality."""

import sys
from unittest.mock import MagicMock

import pytest

# Mock langchain.chains module for tests since it doesn't exist in langchain 1.0
sys.modules['langchain.chains'] = MagicMock()


class TestMainPackageLazyImports:
    """Test lazy imports in main rag_pipeline package."""

    def test_rag_pipeline_lazy_import(self) -> None:
        """Test lazy import of RAGPipeline."""
        from src.rag_pipeline import RAGPipeline

        assert RAGPipeline is not None
        assert RAGPipeline.__name__ == "RAGPipeline"

    def test_rag_pipeline_config_lazy_import(self) -> None:
        """Test lazy import of RAGPipelineConfig."""
        from src.rag_pipeline import RAGPipelineConfig

        assert RAGPipelineConfig is not None
        assert RAGPipelineConfig.__name__ == "RAGPipelineConfig"

    def test_load_config_lazy_import(self) -> None:
        """Test lazy import of load_config."""
        from src.rag_pipeline import load_config

        assert load_config is not None
        assert callable(load_config)

    def test_invalid_attribute(self) -> None:
        """Test that invalid attributes raise AttributeError."""
        import src.rag_pipeline

        with pytest.raises(AttributeError, match="has no attribute 'InvalidAttribute'"):
            _ = src.rag_pipeline.InvalidAttribute  # type: ignore

    def test_all_exports(self) -> None:
        """Test __all__ exports."""
        from src.rag_pipeline import __all__

        assert "RAGPipeline" in __all__
        assert "RAGPipelineConfig" in __all__
        assert "load_config" in __all__


class TestCoreLazyImports:
    """Test lazy imports in core package."""

    def test_embeddings_manager_lazy_import(self) -> None:
        """Test lazy import of EmbeddingsManager."""
        from src.rag_pipeline.core import EmbeddingsManager

        assert EmbeddingsManager is not None
        assert EmbeddingsManager.__name__ == "EmbeddingsManager"

    def test_llm_manager_lazy_import(self) -> None:
        """Test lazy import of LLMManager."""
        from src.rag_pipeline.core import LLMManager

        assert LLMManager is not None
        assert LLMManager.__name__ == "LLMManager"

    def test_rag_pipeline_core_lazy_import(self) -> None:
        """Test lazy import of RAGPipeline from core."""
        from src.rag_pipeline.core import RAGPipeline

        assert RAGPipeline is not None
        assert RAGPipeline.__name__ == "RAGPipeline"

    def test_vector_store_manager_lazy_import(self) -> None:
        """Test lazy import of VectorStoreManager."""
        from src.rag_pipeline.core import VectorStoreManager

        assert VectorStoreManager is not None
        assert VectorStoreManager.__name__ == "VectorStoreManager"

    def test_invalid_core_attribute(self) -> None:
        """Test that invalid attributes raise AttributeError."""
        import src.rag_pipeline.core

        with pytest.raises(AttributeError, match="has no attribute 'InvalidAttribute'"):
            _ = src.rag_pipeline.core.InvalidAttribute  # type: ignore

    def test_core_all_exports(self) -> None:
        """Test __all__ exports."""
        from src.rag_pipeline.core import __all__

        assert "EmbeddingsManager" in __all__
        assert "LLMManager" in __all__
        assert "RAGPipeline" in __all__
        assert "VectorStoreManager" in __all__


class TestUtilsLazyImports:
    """Test lazy imports in utils package."""

    def test_load_config_utils_lazy_import(self) -> None:
        """Test lazy import of load_config from utils."""
        from src.rag_pipeline.utils import load_config

        assert load_config is not None
        assert callable(load_config)

    def test_load_yaml_config_lazy_import(self) -> None:
        """Test lazy import of load_yaml_config."""
        from src.rag_pipeline.utils import load_yaml_config

        assert load_yaml_config is not None
        assert callable(load_yaml_config)

    def test_save_config_lazy_import(self) -> None:
        """Test lazy import of save_config."""
        from src.rag_pipeline.utils import save_config

        assert save_config is not None
        assert callable(save_config)

    def test_document_processor_lazy_import(self) -> None:
        """Test lazy import of DocumentProcessor."""
        from src.rag_pipeline.utils import DocumentProcessor

        assert DocumentProcessor is not None
        assert DocumentProcessor.__name__ == "DocumentProcessor"

    def test_setup_logger_lazy_import(self) -> None:
        """Test lazy import of setup_logger."""
        from src.rag_pipeline.utils import setup_logger

        assert setup_logger is not None
        assert callable(setup_logger)

    def test_invalid_utils_attribute(self) -> None:
        """Test that invalid attributes raise AttributeError."""
        import src.rag_pipeline.utils

        with pytest.raises(AttributeError, match="has no attribute 'InvalidAttribute'"):
            _ = src.rag_pipeline.utils.InvalidAttribute  # type: ignore

    def test_utils_all_exports(self) -> None:
        """Test __all__ exports."""
        from src.rag_pipeline.utils import __all__

        assert "load_config" in __all__
        assert "load_yaml_config" in __all__
        assert "save_config" in __all__
        assert "DocumentProcessor" in __all__
        assert "setup_logger" in __all__
