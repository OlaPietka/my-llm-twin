from my_llm_twin import SYSTEM_PROMPT
from my_llm_twin.parsers.base import Message


def _group_turns(messages: list[Message], separator: str) -> list[tuple[str, str]]:
    """
    Merges consecutive messages from the same sender into one turn,
    joined by the separator. Real DMs are bursty — people send
    3 messages in a row instead of one long one.
    """
    if not messages:
        return []

    turns: list[tuple[str, str]] = []
    current_sender = messages[0].sender
    current_parts: list[str] = [messages[0].content]

    for msg in messages[1:]:
        if msg.sender == current_sender:
            current_parts.append(msg.content)
        else:
            turns.append((current_sender, separator.join(current_parts)))
            current_sender = msg.sender
            current_parts = [msg.content]

    turns.append((current_sender, separator.join(current_parts)))
    return turns


def build_examples(
    segment: list[Message],
    user_name: str,
    separator: str,
    max_context_turns: int,
) -> list[dict]:
    """
    Builds chat-formatted training examples from a conversation segment.
    Each example targets a moment where the user replied — the context
    is the preceding turns (up to max_context_turns).
    """
    turns = _group_turns(segment, separator)
    examples = []

    for i, (sender, content) in enumerate(turns):
        if sender != user_name:
            continue
        if i == 0:
            # user speaks first with no context — nothing to learn from
            continue

        start = max(0, i - max_context_turns)
        context = turns[start:i]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ctx_sender, ctx_content in context:
            role = "assistant" if ctx_sender == user_name else "user"
            messages.append({"role": role, "content": ctx_content})

        messages.append({"role": "assistant", "content": content})
        examples.append({"messages": messages})

    return examples
