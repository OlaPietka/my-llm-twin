"""
Extract message files from chat platform data export zips.

Base class handles the shared zip logic — subclasses just define
which files to grab and what prefix to strip.
"""

import fnmatch
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path


class BaseExtractor(ABC):
    """
    Extracts message files from a platform export zip.

    Subclasses define the glob pattern to match and the prefix to strip
    from extracted paths.
    """

    @property
    @abstractmethod
    def pattern(self) -> str:
        """Glob pattern matching message files inside the zip."""
        ...

    @property
    @abstractmethod
    def strip_prefix(self) -> str:
        """Path prefix to strip when writing extracted files."""
        ...

    def find_message_files(self, zip_path: Path) -> list[str]:
        """
        List all matching message file paths inside the zip.
        """
        with zipfile.ZipFile(zip_path) as zf:
            return [
                name for name in zf.namelist()
                if fnmatch.fnmatch(name, self.pattern)
            ]

    def extract(self, zip_path: Path, output_dir: Path) -> list[Path]:
        """
        Extract matching files from the zip.

        Writes to output_dir with strip_prefix removed from each path.
        Returns list of extracted file paths.
        """
        zip_path = Path(zip_path)
        output_dir = Path(output_dir)

        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        extracted = []

        with zipfile.ZipFile(zip_path) as zf:
            message_files = self.find_message_files(zip_path)

            for name in message_files:
                relative = name[len(self.strip_prefix):]
                dest = output_dir / relative
                dest.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(name) as src:
                    dest.write_bytes(src.read())

                extracted.append(dest)

        return extracted


class FacebookExtractor(BaseExtractor):
    """Extracts message JSONs from a Facebook/Messenger data export."""

    @property
    def pattern(self) -> str:
        return "your_facebook_activity/messages/inbox/*/message_*.json"

    @property
    def strip_prefix(self) -> str:
        return "your_facebook_activity/messages/inbox/"
