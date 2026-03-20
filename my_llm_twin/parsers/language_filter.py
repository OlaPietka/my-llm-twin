"""
Filter conversations by detected language.

Uses langdetect on the combined text of each conversation
to determine the dominant language.
"""

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from .base import Message


def detect_language(messages: list[Message]) -> str | None:
    """
    Detect the dominant language of a conversation.

    Concatenates a sample of message texts and runs langdetect on it.
    Returns ISO 639-1 code (e.g. "pl", "en") or None if detection fails.
    """
    # grab up to 50 messages spread across the conversation for a good sample
    step = max(1, len(messages) // 50)
    sample = " ".join(m.content for m in messages[::step])

    if not sample.strip():
        return None

    try:
        return detect(sample)
    except LangDetectException:
        return None


def filter_by_language(
    conversations: dict[str, list[Message]],
    target_language: str,
) -> dict[str, list[Message]]:
    """
    Keep only conversations where the detected language matches the target.

    Returns a new dict with non-matching conversations removed.
    """
    return {
        title: msgs
        for title, msgs in conversations.items()
        if detect_language(msgs) == target_language
    }
