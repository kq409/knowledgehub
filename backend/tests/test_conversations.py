from routers.conversations import conversation_title_from


def test_conversation_title_from_truncates():
    assert conversation_title_from("  hello   world  ") == "hello world"
    long = "x" * 120
    title = conversation_title_from(long)
    assert len(title) <= 80
    assert title.endswith("…")


def test_conversation_title_from_empty():
    assert conversation_title_from("   ") == "New chat"
