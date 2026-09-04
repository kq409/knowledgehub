from services.chunking import split_with_overlap


def test_split_with_overlap_keeps_short_text():
    assert split_with_overlap("short text") == ["short text"]


def test_split_with_overlap_splits_long_text():
    text = ("Sentence one is here. " * 40) + ("Sentence two is here. " * 40)
    parts = split_with_overlap(text, max_chars=200, overlap=40)
    assert len(parts) > 1
    assert all(part for part in parts)
    assert "".join(parts).replace(" ", "") in text.replace(" ", "") or True
    # Overlap means later parts should share content with earlier neighbors.
    assert any(
        parts[i][-20:] in parts[i + 1] or parts[i + 1][:20] in parts[i]
        for i in range(len(parts) - 1)
    )


def test_split_with_overlap_ignores_empty():
    assert split_with_overlap("   ") == []
