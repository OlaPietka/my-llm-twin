from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class UserNames(BaseModel):
    instagram: str = ""
    messenger: str = ""


class DataPaths(BaseModel):
    raw_dir: str = "data/raw"
    parsed_dir: str = "data/parsed"
    dataset_dir: str = "data/dataset"


class ParsingConfig(BaseModel):
    source: Literal["instagram", "messenger", "both"] = "messenger"
    language: str = "en"  # ISO 639-1 code, e.g. "en", "pl", "de"


class DatasetConfig(BaseModel):
    timeout_hours: float = 3.0  # silence gap to split conversation segments
    max_context_turns: int = 10  # how many preceding turns to include
    separator: str = "<|msg|>"  # joins consecutive messages from same sender


class Config(BaseModel):
    user_names: UserNames
    data: DataPaths = DataPaths()
    parsing: ParsingConfig = ParsingConfig()
    dataset: DatasetConfig = DatasetConfig()


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
