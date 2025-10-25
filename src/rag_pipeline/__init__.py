"""RAG Pipeline package."""

__version__ = "0.1.0"


def __getattr__(name):  # type: ignore
    """Lazy imports for the package."""
    if name == "RAGPipeline":
        from .core.pipeline import RAGPipeline

        return RAGPipeline
    elif name == "RAGPipelineConfig":
        from .models.config import RAGPipelineConfig

        return RAGPipelineConfig
    elif name == "load_config":
        from .utils.config_loader import load_config

        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RAGPipeline",
    "RAGPipelineConfig",
    "load_config",
]
