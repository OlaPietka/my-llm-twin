import json
import zipfile
from pathlib import Path

from my_llm_twin.parsers.messenger import MessengerParser


def make_messenger_zip(tmp_path: Path) -> Path:
    """Create a fake Messenger export zip."""
    zip_path = tmp_path / "export.zip"
    prefix = "your_facebook_activity/messages/inbox"

    files = {
        f"{prefix}/alice_123/message_1.json": {
            "title": "Alice",
            "participants": [{"name": "Alice"}, {"name": "Me"}],
            "messages": [
                {"sender_name": "Alice", "timestamp_ms": 3000, "content": "third"},
                {"sender_name": "Me", "timestamp_ms": 1000, "content": "first"},
                {"sender_name": "Alice", "timestamp_ms": 2000, "content": "second"},
            ],
        },
        # second chunk of the same conversation
        f"{prefix}/alice_123/message_2.json": {
            "title": "Alice",
            "participants": [{"name": "Alice"}, {"name": "Me"}],
            "messages": [
                {"sender_name": "Me", "timestamp_ms": 500, "content": "zeroth"},
            ],
        },
        # different conversation
        f"{prefix}/bob_456/message_1.json": {
            "title": "Bob",
            "participants": [{"name": "Bob"}, {"name": "Me"}],
            "messages": [
                {"sender_name": "Bob", "timestamp_ms": 100, "content": "yo"},
                # photo-only message, no content — should be skipped
                {"sender_name": "Bob", "timestamp_ms": 200, "photos": [{"uri": "pic.jpg"}]},
            ],
        },
    }

    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, json.dumps(content))

    return zip_path


def test_parse_groups_by_conversation(tmp_path):
    parser = MessengerParser()
    result = parser.parse(make_messenger_zip(tmp_path))

    assert set(result.keys()) == {"Alice", "Bob"}


def test_parse_merges_chunks(tmp_path):
    """Messages from message_1.json and message_2.json get combined."""
    parser = MessengerParser()
    result = parser.parse(make_messenger_zip(tmp_path))

    assert len(result["Alice"]) == 4


def test_parse_sorts_by_timestamp(tmp_path):
    parser = MessengerParser()
    result = parser.parse(make_messenger_zip(tmp_path))

    alice_msgs = result["Alice"]
    contents = [m.content for m in alice_msgs]
    assert contents == ["zeroth", "first", "second", "third"]


def test_parse_skips_media_only_messages(tmp_path):
    parser = MessengerParser()
    result = parser.parse(make_messenger_zip(tmp_path))

    # bob had 2 messages but one is photo-only
    assert len(result["Bob"]) == 1
    assert result["Bob"][0].content == "yo"


def test_parse_message_fields(tmp_path):
    parser = MessengerParser()
    result = parser.parse(make_messenger_zip(tmp_path))

    msg = result["Bob"][0]
    assert msg.sender == "Bob"
    assert msg.timestamp == 100
    assert msg.source == "messenger"


def test_parse_skips_group_chats(tmp_path):
    """Group conversations (3+ participants) are excluded."""
    zip_path = tmp_path / "export.zip"
    prefix = "your_facebook_activity/messages/inbox"

    files = {
        # 1:1 — should be included
        f"{prefix}/alice_123/message_1.json": {
            "title": "Alice",
            "participants": [{"name": "Alice"}, {"name": "Me"}],
            "messages": [
                {"sender_name": "Alice", "timestamp_ms": 1000, "content": "hey"},
            ],
        },
        # group chat — should be skipped
        f"{prefix}/group_456/message_1.json": {
            "title": "The gang",
            "participants": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Me"}],
            "messages": [
                {"sender_name": "Alice", "timestamp_ms": 2000, "content": "group msg"},
            ],
        },
    }

    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, json.dumps(content))

    parser = MessengerParser()
    result = parser.parse(zip_path)

    assert "Alice" in result
    assert "The gang" not in result


def test_fix_encoding(tmp_path):
    """Facebook's mojibake gets decoded properly across different languages."""
    zip_path = tmp_path / "export.zip"
    prefix = "your_facebook_activity/messages/inbox"

    # Facebook encodes UTF-8 as Latin-1 escape sequences.
    # To produce test data: "text".encode("utf-8").decode("latin-1")
    cases = {
        # Japanese
        f"{prefix}/yuki_123/message_1.json": {
            "title": "Yuki",
            "participants": [{"name": "Yuki"}, {"name": "Me"}],
            "messages": [{
                "sender_name": "Yuki",
                "timestamp_ms": 1000,
                # "ありがとう" (arigatou)
                "content": "\u00e3\u0081\u0082\u00e3\u0082\u008a\u00e3\u0081\u008c\u00e3\u0081\u00a8\u00e3\u0081\u0086",
            }],
        },
        # German
        f"{prefix}/hans_456/message_1.json": {
            "title": "Hans",
            "participants": [{"name": "Hans"}, {"name": "Me"}],
            "messages": [{
                "sender_name": "Hans",
                "timestamp_ms": 2000,
                # "Straße" (street)
                "content": "Stra\u00c3\u009fe",
            }],
        },
        # emoji
        f"{prefix}/emoji_789/message_1.json": {
            "title": "Emoji crew \u00f0\u009f\u008e\u0089",
            "participants": [{"name": "Someone"}, {"name": "Me"}],
            "messages": [{
                "sender_name": "Someone",
                "timestamp_ms": 3000,
                "content": "party \u00f0\u009f\u008e\u0089",
            }],
        },
    }

    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, data in cases.items():
            zf.writestr(name, json.dumps(data))

    parser = MessengerParser()
    result = parser.parse(zip_path)

    assert result["Yuki"][0].content == "ありがとう"
    assert result["Hans"][0].content == "Straße"
    assert "🎉" in result["Emoji crew 🎉"][0].content
