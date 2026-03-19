import typer

app = typer.Typer(help="Fine-tune an LLM on your DMs to create a digital twin.")


@app.command()
def init():
    """Interactive config setup wizard."""
    typer.echo("Not implemented yet.")


@app.command()
def parse():
    """Parse raw exports into intermediate format."""
    typer.echo("Not implemented yet.")


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
