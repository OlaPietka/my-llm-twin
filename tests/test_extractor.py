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
        # inbox message files we want
        f"{prefix}/inbox/alice_123/message_1.json": {
            "title": "Alice",
            "participants": [{"name": "Alice"}],
            "messages": [{"sender_name": "Alice", "content": "hey"}],
        },
        f"{prefix}/inbox/alice_123/message_2.json": {
            "title": "Alice",
            "participants": [{"name": "Alice"}],
            "messages": [{"sender_name": "Alice", "content": "sup"}],
        },
        # other categories should be ignored
        f"{prefix}/e2ee_cutover/bob_456/message_1.json": {
            "title": "Bob",
            "participants": [{"name": "Bob"}],
            "messages": [{"sender_name": "Bob", "content": "yo"}],
        },
        # metadata files should be ignored
        f"{prefix}/autofill_information.json": {"foo": "bar"},
        # media files should be ignored
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

    assert len(found) == 2
    assert any("alice_123/message_1.json" in f for f in found)
    assert any("alice_123/message_2.json" in f for f in found)
    # other categories, metadata, and media should not be included
    assert not any("e2ee_cutover" in f for f in found)
    assert not any("autofill" in f for f in found)
    assert not any("photos" in f for f in found)


def test_read_messages(tmp_path, extractor):
    zip_path = make_test_zip(tmp_path)

    results = list(extractor.read_messages(zip_path))

    assert len(results) == 2
    # both should be from alice's inbox conversation
    assert all(r["title"] == "Alice" for r in results)
    contents = [r["messages"][0]["content"] for r in results]
    assert "hey" in contents
    assert "sup" in contents


def test_read_messages_is_lazy(tmp_path, extractor):
    """Verify read_messages returns an iterator, not a list."""
    zip_path = make_test_zip(tmp_path)
    result = extractor.read_messages(zip_path)
    # should be a generator, not eagerly loaded
    assert hasattr(result, "__next__")


def test_read_messages_nonexistent_zip(tmp_path, extractor):
    with pytest.raises(FileNotFoundError):
        # need to consume the iterator to trigger the error
        list(extractor.read_messages(tmp_path / "nope.zip"))
