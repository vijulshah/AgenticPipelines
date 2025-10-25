"""Tests for configuration loader utilities."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.rag_pipeline.models.config import RAGPipelineConfig
from src.rag_pipeline.utils.config_loader import (
    deep_merge,
    load_config,
    load_yaml_config,
    substitute_env_vars,
)


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file."""
    config_data = {
        "pipeline_name": "test_pipeline",
        "mongodb": {
            "connection_string": "${MONGODB_URI:mongodb://localhost:27017}",
            "database_name": "test_db",
        },
        "embedding": {
            "api_key": "${OPENAI_API_KEY:test_key}",
        },
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return config_file


class TestLoadYamlConfig:
    """Test YAML config loading."""

    def test_load_valid_yaml(self, temp_config_file: Path) -> None:
        """Test loading valid YAML file."""
        config = load_yaml_config(temp_config_file)
        assert isinstance(config, dict)
        assert config["pipeline_name"] == "test_pipeline"

    def test_load_nonexistent_file(self) -> None:
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config(Path("nonexistent.yaml"))


class TestSubstituteEnvVars:
    """Test environment variable substitution."""

    def test_substitute_with_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test substitution with default values."""
        config = {
            "key": "${TEST_VAR:default_value}",
        }
        result = substitute_env_vars(config)
        assert result["key"] == "default_value"

    def test_substitute_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test substitution with environment variable."""
        monkeypatch.setenv("TEST_VAR", "env_value")
        config = {
            "key": "${TEST_VAR:default_value}",
        }
        result = substitute_env_vars(config)
        assert result["key"] == "env_value"

    def test_substitute_nested(self) -> None:
        """Test substitution in nested dictionaries."""
        config = {
            "level1": {
                "level2": {
                    "key": "${NESTED_VAR:nested_default}",
                }
            }
        }
        result = substitute_env_vars(config)
        assert result["level1"]["level2"]["key"] == "nested_default"

    def test_substitute_in_list(self) -> None:
        """Test substitution in lists."""
        config = {
            "items": ["${VAR1:value1}", "${VAR2:value2}"],
        }
        result = substitute_env_vars(config)
        assert result["items"] == ["value1", "value2"]


class TestDeepMerge:
    """Test deep merge utility."""

    def test_simple_merge(self) -> None:
        """Test simple dictionary merge."""
        base: dict[str, Any] = {"a": 1, "b": 2}
        override: dict[str, Any] = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Test nested dictionary merge."""
        base: dict[str, Any] = {"config": {"a": 1, "b": 2}}
        override: dict[str, Any] = {"config": {"b": 3, "c": 4}}
        result = deep_merge(base, override)
        assert result == {"config": {"a": 1, "b": 3, "c": 4}}


class TestLoadConfig:
    """Test config loading with validation."""

    def test_load_config_from_file(
        self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading config from file."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        config = load_config(temp_config_file)
        assert isinstance(config, RAGPipelineConfig)
        assert config.pipeline_name == "test_pipeline"
        assert config.embedding.api_key == "test_api_key"

    def test_load_config_with_override(self, temp_config_file: Path) -> None:
        """Test loading config with overrides."""
        override = {
            "pipeline_name": "overridden_pipeline",
        }
        config = load_config(temp_config_file, override_config=override)
        assert config.pipeline_name == "overridden_pipeline"
