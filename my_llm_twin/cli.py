from pathlib import Path

import typer

from my_llm_twin.config import load_config

app = typer.Typer(help="Fine-tune an LLM on your DMs to create a digital twin.")

CONFIG_PATH = Path("config.yaml")


@app.command()
def init():
    """Interactive config setup wizard."""
    typer.echo("Not implemented yet.")


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
def build_dataset():
    """Build training dataset from parsed messages."""
    typer.echo("Not implemented yet.")


@app.command()
def train():
    """Fine-tune the model on your dataset."""
    typer.echo("Not implemented yet.")


@app.command()
def chat():
    """Chat with your digital twin."""
    typer.echo("Not implemented yet.")
