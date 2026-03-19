from my_llm_twin.parsers.base import Message


def segment_conversation(
    messages: list[Message],
    timeout_ms: int,
) -> list[list[Message]]:
    """
    Chops a conversation into segments wherever there's a long enough
    silence between messages (longer than timeout_ms).
    """
    if not messages:
        return []

    segments: list[list[Message]] = []
    current: list[Message] = [messages[0]]

    for prev, curr in zip(messages, messages[1:]):
        if curr.timestamp - prev.timestamp > timeout_ms:
            segments.append(current)
            current = []
        current.append(curr)

    segments.append(current)
    return segments
