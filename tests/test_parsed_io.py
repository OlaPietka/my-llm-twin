from pathlib import Path

from my_llm_twin.parsers.base import Message
from my_llm_twin.parsers.io import save_parsed, load_parsed, _safe_filename


def make_conversations():
    return {
        "Alice": [
            Message(sender="Alice", content="hey", timestamp=1000, source="messenger"),
            Message(sender="Me", content="hi", timestamp=2000, source="messenger"),
        ],
        "Bob": [
            Message(sender="Bob", content="yo", timestamp=3000, source="messenger"),
        ],
    }


def test_save_creates_files(tmp_path):
    conversations = make_conversations()
    written = save_parsed(conversations, tmp_path / "parsed")

    assert len(written) == 2
    assert all(p.exists() for p in written)


def test_roundtrip(tmp_path):
    """Save then load should give back the same data."""
    original = make_conversations()
    parsed_dir = tmp_path / "parsed"

    save_parsed(original, parsed_dir)
    loaded = load_parsed(parsed_dir)

    assert set(loaded.keys()) == set(original.keys())
    for title in original:
        assert len(loaded[title]) == len(original[title])
        for orig, load in zip(original[title], loaded[title]):
            assert orig.sender == load.sender
            assert orig.content == load.content
            assert orig.timestamp == load.timestamp
            assert orig.source == load.source


def test_unicode_roundtrip(tmp_path):
    """Non-ASCII titles and content survive save/load."""
    conversations = {
        "Mikołaj": [
            Message(sender="Mikołaj", content="cześć 👋", timestamp=1000, source="messenger"),
        ],
    }

    parsed_dir = tmp_path / "parsed"
    save_parsed(conversations, parsed_dir)
    loaded = load_parsed(parsed_dir)

    assert "Mikołaj" in loaded
    assert loaded["Mikołaj"][0].content == "cześć 👋"


def test_safe_filename():
    assert _safe_filename("Alice") == "Alice"
    assert _safe_filename("GIRLS💁🏼‍♀️") == "GIRLS"
    assert _safe_filename("Oral & Kola 💛🖤") == "Oral _ Kola"
    assert _safe_filename("") == "unnamed"


def test_load_empty_dir(tmp_path):
    parsed_dir = tmp_path / "empty"
    parsed_dir.mkdir()
    assert load_parsed(parsed_dir) == {}
