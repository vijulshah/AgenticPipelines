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

    def test_save_config(self, tmp_path: Path) -> None:
        """Test saving config to file."""
        from src.rag_pipeline.utils.config_loader import save_config
        from src.rag_pipeline.models.config import MongoDBConfig

        config = RAGPipelineConfig(
            pipeline_name="test_save",
            mongodb=MongoDBConfig(connection_string="mongodb://localhost:27017"),
        )

        output_file = tmp_path / "subdir" / "output_config.yaml"
        save_config(config, output_file)

        assert output_file.exists()
        
        # Verify content
        with open(output_file) as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["pipeline_name"] == "test_save"

    def test_substitute_without_env_format(self) -> None:
        """Test substitution with regular strings."""
        config = {
            "regular_string": "no_substitution",
            "number": 42,
            "boolean": True,
        }
        result = substitute_env_vars(config)
        assert result["regular_string"] == "no_substitution"
        assert result["number"] == 42
        assert result["boolean"] is True

    def test_substitute_env_var_without_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test substitution without default value."""
        monkeypatch.setenv("MY_VAR", "my_value")
        config = {
            "key": "${MY_VAR}",
        }
        result = substitute_env_vars(config)
        assert result["key"] == "my_value"

    def test_substitute_missing_env_var_without_default(self) -> None:
        """Test substitution with missing env var and no default."""
        config = {
            "key": "${MISSING_VAR}",
        }
        result = substitute_env_vars(config)
        # Should return the original string when env var is missing
        assert result["key"] == "${MISSING_VAR}"

    def test_load_env_file(self, tmp_path: Path) -> None:
        """Test loading environment variables from .env file."""
        from src.rag_pipeline.utils.config_loader import load_env_file

        # Create a .env file
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_ENV_VAR=test_value\n")

        load_env_file(env_file)
        
        import os
        assert os.getenv("TEST_ENV_VAR") == "test_value"

    def test_load_env_file_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading .env file from current directory."""
        from src.rag_pipeline.utils.config_loader import load_env_file

        # Change to tmp directory
        monkeypatch.chdir(tmp_path)
        
        # Create .env in current directory
        env_file = tmp_path / ".env"
        env_file.write_text("CWD_VAR=cwd_value\n")

        load_env_file()
        
        import os
        assert os.getenv("CWD_VAR") == "cwd_value"

    def test_load_yaml_config_empty_file(self, tmp_path: Path) -> None:
        """Test loading empty YAML file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        result = load_yaml_config(empty_file)
        assert result == {}
