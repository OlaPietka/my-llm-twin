"""
Read message files from chat platform data export zips.

Base class handles the shared zip logic — subclasses just define
which files to match. Reads directly from zip, no disk extraction needed.
"""

import fnmatch
import json
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class BaseExtractor(ABC):
    """
    Reads message files from a platform export zip.

    Subclasses define the glob patterns to match message files.
    """

    @property
    @abstractmethod
    def patterns(self) -> list[str]:
        """Glob patterns matching message files inside the zip."""
        ...

    def _match_names(self, zf: zipfile.ZipFile) -> list[str]:
        """Filter zip entries against our patterns."""
        return [
            name for name in zf.namelist()
            if any(fnmatch.fnmatch(name, p) for p in self.patterns)
        ]

    def find_message_files(self, zip_path: Path) -> list[str]:
        """
        List all matching message file paths inside the zip.
        """
        with zipfile.ZipFile(zip_path) as zf:
            return self._match_names(zf)

    def read_messages(self, zip_path: Path) -> Iterator[dict[str, Any]]:
        """
        Yield parsed JSON dicts from matching files, one at a time.

        Each dict has keys like participants, messages, title, etc.
        Streams from zip — only one file in memory at a time.
        """
        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        with zipfile.ZipFile(zip_path) as zf:
            for name in self._match_names(zf):
                with zf.open(name) as f:
                    yield json.load(f)


class FacebookExtractor(BaseExtractor):
    """Reads message JSONs from a Facebook/Messenger data export."""

    @property
    def patterns(self) -> list[str]:
        return [
            "your_facebook_activity/messages/inbox/*/message_*.json",
            "your_facebook_activity/messages/e2ee_cutover/*/message_*.json",
        ]
