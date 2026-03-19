from my_llm_twin.parsers.base import Message
from my_llm_twin.dataset.segmenter import segment_conversation

MINUTE_MS = 60 * 1000


def make_msg(sender: str, content: str, ts: int) -> Message:
    return Message(sender=sender, content=content, timestamp=ts, source="test")


def test_single_segment():
    msgs = [
        make_msg("A", "hi", 1000),
        make_msg("B", "hey", 2000),
        make_msg("A", "sup", 3000),
    ]
    segments = segment_conversation(msgs, timeout_ms=60 * MINUTE_MS)
    assert len(segments) == 1
    assert len(segments[0]) == 3


def test_split_on_gap():
    msgs = [
        make_msg("A", "hi", 0),
        make_msg("B", "hey", 1000),
        # 2 hour gap
        make_msg("A", "yo", 2 * 60 * MINUTE_MS),
        make_msg("B", "what", 2 * 60 * MINUTE_MS + 1000),
    ]
    segments = segment_conversation(msgs, timeout_ms=60 * MINUTE_MS)
    assert len(segments) == 2
    assert len(segments[0]) == 2
    assert len(segments[1]) == 2


def test_empty_input():
    segments = segment_conversation([], timeout_ms=60 * MINUTE_MS)
    assert segments == []


def test_single_message():
    msgs = [make_msg("A", "hi", 1000)]
    segments = segment_conversation(msgs, timeout_ms=60 * MINUTE_MS)
    assert len(segments) == 1
    assert len(segments[0]) == 1
