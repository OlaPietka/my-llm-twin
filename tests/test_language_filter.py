from my_llm_twin.parsers.base import Message
from my_llm_twin.parsers.language_filter import detect_language, filter_by_language


def _make_messages(texts: list[str], sender: str = "Someone") -> list[Message]:
    return [
        Message(sender=sender, content=t, timestamp=i * 1000, source="messenger")
        for i, t in enumerate(texts)
    ]


def test_detect_polish():
    msgs = _make_messages([
        "Cześć, jak się masz?",
        "Wszystko dobrze, dzięki za pytanie",
        "Fajnie, co robisz dzisiaj?",
        "Idę na zakupy, a potem do kina",
        "Super, baw się dobrze!",
    ])
    assert detect_language(msgs) == "pl"


def test_detect_english():
    msgs = _make_messages([
        "Hey, how are you doing?",
        "Pretty good, thanks for asking",
        "Nice, what are you up to today?",
        "Going shopping and then to the movies",
        "Cool, have fun!",
    ])
    assert detect_language(msgs) == "en"


def test_detect_empty():
    assert detect_language([]) is None


def test_filter_by_language():
    conversations = {
        "Polish friend": _make_messages([
            "Cześć, jak się masz?",
            "Wszystko dobrze, dzięki",
            "Co słychać u ciebie?",
            "Mam się świetnie, dzięki za pytanie",
        ]),
        "English friend": _make_messages([
            "Hey, how are you doing today?",
            "Pretty good, thanks for asking me",
            "What are you up to this weekend?",
            "Going to the beach with friends",
        ]),
        "Another Polish friend": _make_messages([
            "Hej, co robisz dzisiaj wieczorem?",
            "Idę na zakupy, a potem do kina",
            "Może pójdziemy razem na kawę?",
            "Jasne, spotkajmy się o piątej",
        ]),
    }

    result = filter_by_language(conversations, "pl")

    assert "Polish friend" in result
    assert "Another Polish friend" in result
    assert "English friend" not in result
