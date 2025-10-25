"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from src.rag_pipeline.models.config import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    MongoDBConfig,
    RAGPipelineConfig,
    RetrieverConfig,
)


class TestMongoDBConfig:
    """Test MongoDB configuration model."""

    def test_valid_mongodb_config(self) -> None:
        """Test valid MongoDB configuration."""
        config = MongoDBConfig(
            connection_string="mongodb://localhost:27017",
            database_name="test_db",
            collection_name="test_collection",
        )
        assert config.connection_string == "mongodb://localhost:27017"
        assert config.database_name == "test_db"

    def test_mongodb_srv_connection(self) -> None:
        """Test MongoDB+SRV connection string."""
        config = MongoDBConfig(
            connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
        )
        assert config.connection_string.startswith("mongodb+srv://")

    def test_invalid_connection_string(self) -> None:
        """Test invalid MongoDB connection string."""
        with pytest.raises(ValidationError):
            MongoDBConfig(connection_string="invalid://localhost:27017")


class TestEmbeddingConfig:
    """Test embedding configuration model."""

    def test_default_embedding_config(self) -> None:
        """Test default embedding configuration."""
        config = EmbeddingConfig()
        assert config.provider == "openai"
        assert config.model_name == "text-embedding-3-small"
        assert config.dimensions == 1536

    def test_custom_embedding_config(self) -> None:
        """Test custom embedding configuration."""
        config = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-large",
            dimensions=3072,
        )
        assert config.dimensions == 3072


class TestLLMConfig:
    """Test LLM configuration model."""

    def test_default_llm_config(self) -> None:
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.7

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            LLMConfig(temperature=3.0)  # Too high

        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)  # Too low


class TestRetrieverConfig:
    """Test retriever configuration model."""

    def test_default_retriever_config(self) -> None:
        """Test default retriever configuration."""
        config = RetrieverConfig()
        assert config.search_type == "similarity"
        assert config.k == 4

    def test_custom_retriever_config(self) -> None:
        """Test custom retriever configuration."""
        config = RetrieverConfig(
            search_type="mmr",
            k=10,
            score_threshold=0.8,
        )
        assert config.k == 10
        assert config.score_threshold == 0.8


class TestChunkingConfig:
    """Test chunking configuration model."""

    def test_default_chunking_config(self) -> None:
        """Test default chunking configuration."""
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_chunk_overlap_validation(self) -> None:
        """Test chunk overlap validation."""
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=500, chunk_overlap=600)


class TestRAGPipelineConfig:
    """Test RAG pipeline configuration model."""

    def test_default_config(self) -> None:
        """Test default RAG pipeline configuration."""
        config = RAGPipelineConfig(
            mongodb=MongoDBConfig(connection_string="mongodb://localhost:27017")
        )
        assert config.pipeline_name == "default_rag_pipeline"
        assert config.vector_store_type == "mongodb"
        assert config.enable_logging is True

    def test_full_custom_config(self) -> None:
        """Test fully custom RAG pipeline configuration."""
        config = RAGPipelineConfig(
            pipeline_name="test_pipeline",
            mongodb=MongoDBConfig(
                connection_string="mongodb://localhost:27017",
                database_name="test_db",
            ),
            embedding=EmbeddingConfig(
                model_name="text-embedding-3-large",
                dimensions=3072,
            ),
            llm=LLMConfig(
                model_name="gpt-4",
                temperature=0.5,
            ),
            retriever=RetrieverConfig(k=5),
            chunking=ChunkingConfig(chunk_size=800),
        )
        assert config.pipeline_name == "test_pipeline"
        assert config.embedding.dimensions == 3072
        assert config.llm.temperature == 0.5
