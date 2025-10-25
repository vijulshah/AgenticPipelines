"""Configuration utilities for loading and validating configs."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from ..models.config import RAGPipelineConfig


def load_env_file(env_path: Optional[Path] = None) -> None:
    """Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, searches in current directory.
    """
    if env_path is None:
        env_path = Path.cwd() / ".env"

    if env_path.exists():
        load_dotenv(env_path)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration data.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return config_data or {}


def substitute_env_vars(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively substitute environment variables in config.

    Replaces strings in format ${ENV_VAR} with environment variable values.

    Args:
        config_data: Configuration dictionary.

    Returns:
        Configuration dictionary with substituted values.
    """

    def _substitute(value: Any) -> Any:
        if isinstance(value, str):
            # Handle ${ENV_VAR} or ${ENV_VAR:default_value} format
            if value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                # Handle default values
                if ":" in env_var:
                    var_name, default = env_var.split(":", 1)
                    return os.getenv(var_name.strip(), default.strip())
                return os.getenv(env_var, value)
            return value
        elif isinstance(value, dict):
            return {k: _substitute(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_substitute(item) for item in value]
        return value

    return _substitute(config_data)


def load_config(
    config_path: Path,
    env_path: Optional[Path] = None,
    override_config: Optional[Dict[str, Any]] = None,
) -> RAGPipelineConfig:
    """Load and validate RAG pipeline configuration.

    Args:
        config_path: Path to YAML configuration file.
        env_path: Path to .env file for environment variables.
        override_config: Optional dictionary to override config values.

    Returns:
        Validated RAGPipelineConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If configuration validation fails.
    """
    # Load environment variables
    load_env_file(env_path)

    # Load YAML configuration
    config_data = load_yaml_config(config_path)

    # Substitute environment variables
    config_data = substitute_env_vars(config_data)

    # Apply overrides if provided
    if override_config:
        config_data = deep_merge(config_data, override_config)

    # Validate and create Pydantic model
    return RAGPipelineConfig(**config_data)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Override dictionary.

    Returns:
        Merged dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: RAGPipelineConfig, output_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: RAGPipelineConfig instance.
        output_path: Path to save YAML file.
    """
    # Convert Pydantic model to dict
    config_dict = config.model_dump(mode="python", exclude_none=True)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to YAML
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
