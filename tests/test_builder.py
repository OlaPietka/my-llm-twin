from my_llm_twin.parsers.base import Message
from my_llm_twin.dataset.builder import build_examples

SEP = "<|msg|>"


def make_msg(sender: str, content: str, ts: int) -> Message:
    return Message(sender=sender, content=content, timestamp=ts, source="test")


def test_simple_exchange():
    segment = [
        make_msg("friend", "hey", 1000),
        make_msg("me", "yo", 2000),
    ]
    examples = build_examples(segment, user_name="me", separator=SEP, max_context_turns=5)
    assert len(examples) == 1
    assert examples[0]["messages"][-1]["role"] == "assistant"
    assert examples[0]["messages"][-1]["content"] == "yo"
    assert examples[0]["messages"][-2]["role"] == "user"
    assert examples[0]["messages"][-2]["content"] == "hey"


def test_multi_message_grouping():
    segment = [
        make_msg("friend", "hey", 1000),
        make_msg("friend", "whats up?", 2000),
        make_msg("me", "not much", 3000),
        make_msg("me", "chilling", 4000),
        make_msg("me", "you?", 5000),
    ]
    examples = build_examples(segment, user_name="me", separator=SEP, max_context_turns=5)
    assert len(examples) == 1
    ex = examples[0]["messages"]
    assert ex[-2]["content"] == f"hey{SEP}whats up?"
    assert ex[-1]["content"] == f"not much{SEP}chilling{SEP}you?"


def test_multiple_exchanges():
    segment = [
        make_msg("friend", "hey", 1000),
        make_msg("me", "yo", 2000),
        make_msg("friend", "what are you doing?", 3000),
        make_msg("me", "nothing", 4000),
    ]
    examples = build_examples(segment, user_name="me", separator=SEP, max_context_turns=5)
    assert len(examples) == 2


def test_no_user_reply():
    segment = [
        make_msg("friend", "hey", 1000),
        make_msg("friend", "hello?", 2000),
    ]
    examples = build_examples(segment, user_name="me", separator=SEP, max_context_turns=5)
    assert len(examples) == 0


def test_context_window_limit():
    segment = [
        make_msg("friend", "msg1", 1000),
        make_msg("me", "r1", 2000),
        make_msg("friend", "msg2", 3000),
        make_msg("me", "r2", 4000),
        make_msg("friend", "msg3", 5000),
        make_msg("me", "r3", 6000),
        make_msg("friend", "msg4", 7000),
        make_msg("me", "r4", 8000),
    ]
    examples = build_examples(segment, user_name="me", separator=SEP, max_context_turns=2)
    last = examples[-1]["messages"]
    # should have at most 2 context turns + 1 assistant (excluding system)
    non_system = [m for m in last if m["role"] != "system"]
    assert len(non_system) <= 3
