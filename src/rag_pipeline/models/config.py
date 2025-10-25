"""Pydantic models for RAG pipeline configuration."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VectorStoreType(str, Enum):
    """Supported vector store types."""

    MONGODB = "mongodb"
    CHROMA = "chroma"
    PINECONE = "pinecone"


class MongoDBConfig(BaseModel):
    """MongoDB configuration model."""

    connection_string: str = Field(
        ...,
        description="MongoDB connection string",
        examples=["mongodb://localhost:27017"],
    )
    database_name: str = Field(
        default="rag_database",
        description="Database name for vector storage",
    )
    collection_name: str = Field(
        default="vector_store",
        description="Collection name for vectors",
    )
    index_name: str = Field(
        default="vector_index",
        description="Index name for vector search",
    )

    @field_validator("connection_string")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        """Validate MongoDB connection string format."""
        if not v.startswith("mongodb://") and not v.startswith("mongodb+srv://"):
            raise ValueError(
                "Connection string must start with 'mongodb://' or 'mongodb+srv://'"
            )
        return v


class EmbeddingConfig(BaseModel):
    """Embedding configuration model."""

    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="Embedding provider to use",
    )
    model_name: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    dimensions: Optional[int] = Field(
        default=1536,
        description="Embedding dimensions",
        ge=1,
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the embedding provider",
    )


class LLMConfig(BaseModel):
    """LLM configuration model."""

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="LLM model name",
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for text generation",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum tokens in response",
        ge=1,
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the LLM provider",
    )


class RetrieverConfig(BaseModel):
    """Retriever configuration model."""

    search_type: str = Field(
        default="similarity",
        description="Type of search to perform",
    )
    k: int = Field(
        default=4,
        description="Number of documents to retrieve",
        ge=1,
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum relevance score threshold",
        ge=0.0,
        le=1.0,
    )


class ChunkingConfig(BaseModel):
    """Document chunking configuration model."""

    chunk_size: int = Field(
        default=1000,
        description="Size of each chunk in characters",
        ge=100,
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks",
        ge=0,
    )
    separator: str = Field(
        default="\n\n",
        description="Separator for splitting documents",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: Any) -> int:
        """Validate chunk overlap is less than chunk size."""
        # Note: info.data contains already validated fields
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class RAGPipelineConfig(BaseModel):
    """Main RAG pipeline configuration model."""

    pipeline_name: str = Field(
        default="default_rag_pipeline",
        description="Name of the RAG pipeline",
    )
    vector_store_type: VectorStoreType = Field(
        default=VectorStoreType.MONGODB,
        description="Type of vector store to use",
    )
    mongodb: MongoDBConfig = Field(
        default_factory=MongoDBConfig,
        description="MongoDB configuration",
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding configuration",
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration",
    )
    retriever: RetrieverConfig = Field(
        default_factory=RetrieverConfig,
        description="Retriever configuration",
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="Chunking configuration",
    )
    enable_logging: bool = Field(
        default=True,
        description="Enable logging",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
