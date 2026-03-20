"""
Parser for Facebook Messenger data exports.

Reads message JSONs from a zip via FacebookExtractor and converts
them into Message dataclasses grouped by conversation.
"""

from collections import defaultdict
from pathlib import Path

from .base import BaseParser, Message
from .extractor import FacebookExtractor


class MessengerParser(BaseParser):
    def __init__(self):
        self._extractor = FacebookExtractor()

    @staticmethod
    def _fix_encoding(text: str) -> str:
        """
        Fix Facebook's broken encoding.

        Facebook exports UTF-8 text as Latin-1 escape sequences,
        so e.g. "ż" becomes "Å¼". Re-encoding fixes it.
        """
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return text

    def parse(self, zip_path: Path) -> dict[str, list[Message]]:
        """
        Parse a Facebook Messenger export zip.

        Returns conversations grouped by title, each sorted by timestamp.
        Skips messages without text content (photos, videos, stickers, etc.).
        """
        conversations: dict[str, list[Message]] = defaultdict(list)

        for chunk in self._extractor.read_messages(zip_path):
            title = self._fix_encoding(chunk["title"])

            for raw_msg in chunk.get("messages", []):
                content = raw_msg.get("content")
                if not content:
                    continue

                msg = Message(
                    sender=self._fix_encoding(raw_msg["sender_name"]),
                    content=self._fix_encoding(content),
                    timestamp=raw_msg["timestamp_ms"],
                    source="messenger",
                )
                conversations[title].append(msg)

        # sort each conversation by timestamp
        for msgs in conversations.values():
            msgs.sort(key=lambda m: m.timestamp)

        return dict(conversations)
