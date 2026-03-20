from pathlib import Path
from typing import Optional

import typer

from my_llm_twin.config import load_config

app = typer.Typer(help="Fine-tune an LLM on your DMs to create a digital twin.")

CONFIG_PATH = Path("config.yaml")


@app.command()
def init(
    config: Path = typer.Option(CONFIG_PATH, help="Path to write config.yaml"),
):
    """Interactive config setup wizard."""
    import yaml

    if config.exists():
        overwrite = typer.confirm(f"{config} already exists. Overwrite?", default=False)
        if not overwrite:
            raise typer.Exit()

    messenger_name = typer.prompt("What's your Messenger display name?")
    language = typer.prompt("Target language (ISO code)", default="en")

    cfg = {
        "user_names": {
            "messenger": messenger_name,
        },
        "parsing": {
            "source": "messenger",
            "language": language,
        },
    }

    config.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
    typer.echo(f"Config saved to {config}")


@app.command()
def parse(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
):
    """Parse raw exports into intermediate format."""
    from my_llm_twin.parsers.io import save_parsed
    from my_llm_twin.parsers.language_filter import filter_by_language
    from my_llm_twin.parsers.messenger import MessengerParser

    cfg = load_config(config)
    raw_dir = Path(cfg.data.raw_dir)
    parsed_dir = Path(cfg.data.parsed_dir)

    # find zip files in raw_dir
    zips = list(raw_dir.glob("*.zip"))
    if not zips:
        typer.echo(f"No zip files found in {raw_dir}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(zips)} zip file(s) in {raw_dir}")

    # parse each zip
    parser = MessengerParser()
    all_conversations: dict[str, list] = {}

    for zip_path in zips:
        typer.echo(f"Parsing {zip_path.name}...")
        conversations = parser.parse(zip_path)
        typer.echo(f"  {len(conversations)} 1:1 conversations, "
                   f"{sum(len(m) for m in conversations.values())} messages")
        all_conversations.update(conversations)

    # filter by language
    language = cfg.parsing.language
    typer.echo(f"Filtering for language: {language}")
    filtered = filter_by_language(all_conversations, language)
    removed = len(all_conversations) - len(filtered)
    typer.echo(f"  Kept {len(filtered)} conversations, removed {removed}")

    # save
    written = save_parsed(filtered, parsed_dir)
    total_msgs = sum(len(m) for m in filtered.values())
    typer.echo(f"Saved {len(written)} files to {parsed_dir}/ ({total_msgs} messages)")


@app.command()
def build_dataset(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
):
    """Build training dataset from parsed messages."""
    import json
    import random

    from my_llm_twin.dataset.builder import build_examples
    from my_llm_twin.dataset.segmenter import segment_conversation
    from my_llm_twin.parsers.io import load_parsed

    cfg = load_config(config)
    parsed_dir = Path(cfg.data.parsed_dir)
    dataset_dir = Path(cfg.data.dataset_dir)

    conversations = load_parsed(parsed_dir)
    if not conversations:
        typer.echo(f"No parsed conversations found in {parsed_dir}")
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(conversations)} conversations from {parsed_dir}")

    user_name = cfg.user_names.messenger
    if not user_name:
        typer.echo("No messenger user name set in config")
        raise typer.Exit(1)

    timeout_ms = int(cfg.dataset.timeout_hours * 3600 * 1000)
    separator = cfg.dataset.separator
    max_context_turns = cfg.dataset.max_context_turns

    # split conversations into train/val (conversation-level, not example-level)
    titles = sorted(conversations.keys())
    random.seed(42)
    random.shuffle(titles)
    split_idx = int(len(titles) * cfg.dataset.train_val_split)
    train_titles = set(titles[:split_idx])
    val_titles = set(titles[split_idx:])

    train_examples = []
    val_examples = []
    total_segments = 0

    for title, messages in conversations.items():
        segments = segment_conversation(messages, timeout_ms)
        total_segments += len(segments)

        for segment in segments:
            examples = build_examples(segment, user_name, separator, max_context_turns)
            if title in train_titles:
                train_examples.extend(examples)
            else:
                val_examples.extend(examples)

    typer.echo(f"  {total_segments} segments")
    typer.echo(f"  {len(train_examples)} train / {len(val_examples)} val examples "
               f"({len(train_titles)} / {len(val_titles)} conversations)")

    # save as JSONL
    dataset_dir.mkdir(parents=True, exist_ok=True)

    def _write_jsonl(path: Path, examples: list[dict]):
        with open(path, "w") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    _write_jsonl(dataset_dir / "train.jsonl", train_examples)
    _write_jsonl(dataset_dir / "val.jsonl", val_examples)

    typer.echo(f"Saved to {dataset_dir}/")


@app.command()
def train(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
):
    """Fine-tune the model on your dataset."""
    import logging

    from my_llm_twin.training.trainer import run_training

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(config)

    typer.echo(f"Training with base model: {cfg.training.base_model}")
    typer.echo(f"Dataset dir: {cfg.data.dataset_dir}")

    run_training(
        training_config=cfg.training,
        dataset_config=cfg.dataset,
        dataset_dir=Path(cfg.data.dataset_dir),
    )


@app.command()
def chat(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
    temperature: Optional[float] = typer.Option(
        None, help="Sampling temperature (default: from config)"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help="Max new tokens to generate (default: from config)"
    ),
):
    """Chat with your digital twin in the terminal."""
    from my_llm_twin.chat.engine import ChatEngine

    cfg = load_config(config)
    temp = temperature if temperature is not None else cfg.chat.temperature
    max_new = max_tokens if max_tokens is not None else cfg.chat.max_new_tokens

    typer.echo(f"Loading model from {cfg.training.output_dir}...")
    engine = ChatEngine(
        model_dir=cfg.training.output_dir,
        separator=cfg.dataset.separator,
        temperature=temp,
        max_new_tokens=max_new,
        top_p=cfg.chat.top_p,
    )
    typer.echo("Ready! Type your message, /reset to clear history, or /quit to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nBye!")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            typer.echo("Bye!")
            break
        if user_input == "/reset":
            engine.reset()
            typer.echo("History cleared.\n")
            continue

        parts = engine.generate(user_input)
        for part in parts:
            typer.echo(f"Twin: {part}")
        typer.echo()
