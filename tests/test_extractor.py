import json
import zipfile
from pathlib import Path

import pytest

from my_llm_twin.parsers.extractor import FacebookExtractor


def make_test_zip(tmp_path: Path) -> Path:
    """Create a fake Facebook export zip with a couple of conversations."""
    zip_path = tmp_path / "export.zip"
    prefix = "your_facebook_activity/messages"

    files = {
        # actual message files we want extracted
        f"{prefix}/inbox/alice_123/message_1.json": {
            "participants": [{"name": "Alice"}],
            "messages": [{"sender_name": "Alice", "content": "hey"}],
        },
        f"{prefix}/inbox/alice_123/message_2.json": {
            "participants": [{"name": "Alice"}],
            "messages": [{"sender_name": "Alice", "content": "sup"}],
        },
        f"{prefix}/e2ee_cutover/bob_456/message_1.json": {
            "participants": [{"name": "Bob"}],
            "messages": [{"sender_name": "Bob", "content": "yo"}],
        },
        # metadata files that should be ignored
        f"{prefix}/autofill_information.json": {"foo": "bar"},
        # media files that should be ignored
        f"{prefix}/inbox/alice_123/photos/pic.jpg": b"fake image",
    }

    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            if isinstance(content, bytes):
                zf.writestr(name, content)
            else:
                zf.writestr(name, json.dumps(content))

    return zip_path


@pytest.fixture
def extractor():
    return FacebookExtractor()


def test_find_message_files(tmp_path, extractor):
    zip_path = make_test_zip(tmp_path)
    found = extractor.find_message_files(zip_path)

    assert len(found) == 3
    assert any("alice_123/message_1.json" in f for f in found)
    assert any("alice_123/message_2.json" in f for f in found)
    assert any("bob_456/message_1.json" in f for f in found)
    # metadata and media should not be included
    assert not any("autofill" in f for f in found)
    assert not any("photos" in f for f in found)


def test_extract(tmp_path, extractor):
    zip_path = make_test_zip(tmp_path)
    output_dir = tmp_path / "extracted"

    extracted = extractor.extract(zip_path, output_dir)

    assert len(extracted) == 3

    # check files land in the right place with prefix stripped
    alice_msg = output_dir / "inbox" / "alice_123" / "message_1.json"
    assert alice_msg.exists()

    data = json.loads(alice_msg.read_text())
    assert data["messages"][0]["content"] == "hey"

    # e2ee_cutover category preserved
    bob_msg = output_dir / "e2ee_cutover" / "bob_456" / "message_1.json"
    assert bob_msg.exists()


def test_extract_nonexistent_zip(tmp_path, extractor):
    with pytest.raises(FileNotFoundError):
        extractor.extract(tmp_path / "nope.zip", tmp_path / "out")
