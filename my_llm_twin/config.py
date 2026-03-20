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
    train_val_split: float = 0.9  # fraction of conversations for training


class TrainingConfig(BaseModel):
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "models/my-twin"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    max_seq_length: int = 512
    precision: str = "bf16"


class ChatConfig(BaseModel):
    temperature: float = 0.7
    max_new_tokens: int = 256
    top_p: float = 0.9


class Config(BaseModel):
    user_names: UserNames
    data: DataPaths = DataPaths()
    parsing: ParsingConfig = ParsingConfig()
    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    chat: ChatConfig = ChatConfig()


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
