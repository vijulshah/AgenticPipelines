"""Core RAG pipeline components."""

__all__ = [
    "EmbeddingsManager",
    "LLMManager",
    "RAGPipeline",
    "VectorStoreManager",
]


def __getattr__(name):  # type: ignore
    """Lazy imports for core components."""
    if name == "EmbeddingsManager":
        from .embeddings import EmbeddingsManager

        return EmbeddingsManager
    elif name == "LLMManager":
        from .llm import LLMManager

        return LLMManager
    elif name == "RAGPipeline":
        from .pipeline import RAGPipeline

        return RAGPipeline
    elif name == "VectorStoreManager":
        from .vector_store import VectorStoreManager

        return VectorStoreManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
