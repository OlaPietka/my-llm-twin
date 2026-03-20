import pytest
import yaml
from pathlib import Path
from my_llm_twin.config import Config, load_config


def test_load_from_yaml(tmp_path):
    config_data = {
        "user_names": {"instagram": "test_user", "messenger": "Test User"},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)
    assert config.user_names.instagram == "test_user"
    assert config.user_names.messenger == "Test User"


def test_defaults_applied():
    config = Config(
        user_names={"instagram": "test", "messenger": "Test"},
    )
    assert config.data.raw_dir == "data/raw"
    assert config.parsing.source == "messenger"
    assert config.parsing.language == "en"


def test_missing_user_names_raises():
    with pytest.raises(ValueError):
        Config()


def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))
