"""
Save and load parsed conversations as JSON files.

One file per conversation in the output directory,
each containing a list of message dicts.
"""

import json
from pathlib import Path

from .base import Message


def save_parsed(
    conversations: dict[str, list[Message]],
    output_dir: Path,
) -> list[Path]:
    """
    Save parsed conversations to JSON files.

    Each conversation gets its own file in output_dir, named after the
    conversation title (sanitized for filesystem safety).
    Returns list of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for title, messages in conversations.items():
        data = {
            "title": title,
            "messages": [
                {
                    "sender": m.sender,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "source": m.source,
                }
                for m in messages
            ],
        }

        filename = _safe_filename(title) + ".json"
        path = output_dir / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        written.append(path)

    return written


def load_parsed(parsed_dir: Path) -> dict[str, list[Message]]:
    """
    Load all parsed conversation JSON files from a directory.

    Returns a dict mapping conversation title to list of Messages.
    """
    parsed_dir = Path(parsed_dir)
    conversations = {}

    for path in sorted(parsed_dir.glob("*.json")):
        data = json.loads(path.read_text())
        title = data["title"]
        messages = [
            Message(
                sender=m["sender"],
                content=m["content"],
                timestamp=m["timestamp"],
                source=m["source"],
            )
            for m in data["messages"]
        ]
        conversations[title] = messages

    return conversations


def _safe_filename(title: str) -> str:
    """
    Turn a conversation title into a safe filename.

    Replaces anything that's not alphanumeric, space, or hyphen with underscore.
    """
    safe = ""
    for ch in title:
        if ch.isalnum() or ch in (" ", "-"):
            safe += ch
        else:
            safe += "_"
    # collapse multiple underscores, strip, truncate
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_ ")[:100] or "unnamed"
