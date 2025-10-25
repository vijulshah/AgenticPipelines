"""Utility modules for RAG pipeline."""

__all__ = [
    "load_config",
    "load_yaml_config",
    "save_config",
    "DocumentProcessor",
    "setup_logger",
]


def __getattr__(name):  # type: ignore
    """Lazy imports for utilities."""
    if name in ("load_config", "load_yaml_config", "save_config"):
        from .config_loader import load_config, load_yaml_config, save_config

        if name == "load_config":
            return load_config
        elif name == "load_yaml_config":
            return load_yaml_config
        elif name == "save_config":
            return save_config
    elif name == "DocumentProcessor":
        from .document_processor import DocumentProcessor

        return DocumentProcessor
    elif name == "setup_logger":
        from .logger import setup_logger

        return setup_logger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
