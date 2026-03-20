from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Message:
    sender: str
    content: str
    timestamp: int  # milliseconds since epoch
    source: str  # "instagram", "messenger", etc.


class BaseParser(ABC):
    @abstractmethod
    def parse(self, zip_path: Path) -> dict[str, list[Message]]:
        """
        Parse a platform export zip.

        Returns a dict mapping conversation title to list of Messages,
        sorted by timestamp.
        """
        ...
