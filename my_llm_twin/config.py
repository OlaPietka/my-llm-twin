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
    source: Literal["instagram", "messenger", "both"] = "both"


class Config(BaseModel):
    user_names: UserNames
    data: DataPaths = DataPaths()
    parsing: ParsingConfig = ParsingConfig()


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
