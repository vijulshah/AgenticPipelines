"""Integration tests for RAG pipeline structure."""

import pytest

from src.rag_pipeline.models.config import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    MongoDBConfig,
    RAGPipelineConfig,
    RetrieverConfig,
)


class TestRAGPipelineStructure:
    """Test RAG pipeline structure and configuration."""

    def test_complete_config_creation(self) -> None:
        """Test creating a complete RAG pipeline configuration."""
        config = RAGPipelineConfig(
            pipeline_name="test_pipeline",
            mongodb=MongoDBConfig(
                connection_string="mongodb://localhost:27017",
                database_name="test_db",
                collection_name="test_collection",
                index_name="test_index",
            ),
            embedding=EmbeddingConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                dimensions=1536,
            ),
            llm=LLMConfig(
                provider="openai",
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000,
            ),
            retriever=RetrieverConfig(
                search_type="similarity",
                k=4,
            ),
            chunking=ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=200,
            ),
            enable_logging=True,
            log_level="INFO",
        )

        # Verify configuration
        assert config.pipeline_name == "test_pipeline"
        assert config.mongodb.database_name == "test_db"
        assert config.embedding.model_name == "text-embedding-3-small"
        assert config.llm.model_name == "gpt-4o-mini"
        assert config.retriever.k == 4
        assert config.chunking.chunk_size == 1000

    def test_config_serialization(self) -> None:
        """Test configuration serialization to dict."""
        config = RAGPipelineConfig(
            mongodb=MongoDBConfig(connection_string="mongodb://localhost:27017")
        )

        # Serialize to dict
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert "pipeline_name" in config_dict
        assert "mongodb" in config_dict
        assert "embedding" in config_dict
        assert "llm" in config_dict

    def test_config_with_defaults(self) -> None:
        """Test that default values are properly set."""
        config = RAGPipelineConfig(
            mongodb=MongoDBConfig(connection_string="mongodb://localhost:27017")
        )

        # Check defaults
        assert config.pipeline_name == "default_rag_pipeline"
        assert config.embedding.provider == "openai"
        assert config.llm.provider == "openai"
        assert config.retriever.k == 4
        assert config.chunking.chunk_size == 1000
        assert config.enable_logging is True
        assert config.log_level == "INFO"

    def test_lazy_imports(self) -> None:
        """Test that lazy imports work correctly."""
        # Import from main package
        from src.rag_pipeline import RAGPipelineConfig, load_config

        assert RAGPipelineConfig is not None
        assert load_config is not None

        # Import from models
        from src.rag_pipeline.models import (
            ChunkingConfig,
            EmbeddingConfig,
            LLMConfig,
            MongoDBConfig,
        )

        assert ChunkingConfig is not None
        assert EmbeddingConfig is not None
        assert LLMConfig is not None
        assert MongoDBConfig is not None

    def test_config_validation_errors(self) -> None:
        """Test that validation errors are raised for invalid configs."""
        # Invalid MongoDB connection string
        with pytest.raises(Exception):  # ValidationError from pydantic
            MongoDBConfig(connection_string="invalid://localhost")

        # Invalid temperature
        with pytest.raises(Exception):
            LLMConfig(temperature=3.0)

        # Invalid chunk overlap
        with pytest.raises(Exception):
            ChunkingConfig(chunk_size=500, chunk_overlap=600)


class TestDocumentationExists:
    """Test that all documentation files exist."""

    def test_readme_exists(self) -> None:
        """Test that README.md exists."""
        from pathlib import Path

        readme = Path("README.md")
        assert readme.exists()

    def test_quickstart_exists(self) -> None:
        """Test that QUICKSTART.md exists."""
        from pathlib import Path

        quickstart = Path("QUICKSTART.md")
        assert quickstart.exists()

    def test_implementation_exists(self) -> None:
        """Test that IMPLEMENTATION.md exists."""
        from pathlib import Path

        implementation = Path("IMPLEMENTATION.md")
        assert implementation.exists()

    def test_changelog_exists(self) -> None:
        """Test that CHANGELOG.md exists."""
        from pathlib import Path

        changelog = Path("CHANGELOG.md")
        assert changelog.exists()

    def test_example_config_exists(self) -> None:
        """Test that example config exists."""
        from pathlib import Path

        config = Path("configs/default_config.yaml")
        assert config.exists()

    def test_env_example_exists(self) -> None:
        """Test that .env.example exists."""
        from pathlib import Path

        env_example = Path(".env.example")
        assert env_example.exists()


class TestExamplesExist:
    """Test that example scripts exist."""

    def test_basic_usage_example(self) -> None:
        """Test basic usage example exists."""
        from pathlib import Path

        example = Path("examples/basic_usage.py")
        assert example.exists()

    def test_document_processing_example(self) -> None:
        """Test document processing example exists."""
        from pathlib import Path

        example = Path("examples/document_processing.py")
        assert example.exists()

    def test_custom_config_example(self) -> None:
        """Test custom config example exists."""
        from pathlib import Path

        example = Path("examples/custom_config.py")
        assert example.exists()
