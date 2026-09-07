from unittest.mock import MagicMock, patch

from services.rerank import rerank_enabled, rerank_texts


def test_rerank_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    assert rerank_enabled() is False


def test_rerank_texts_identity_when_ranker_missing():
    with patch("services.rerank._get_ranker", return_value=None):
        assert rerank_texts("query", ["a", "b"]) == [(0, 0.0), (1, 0.0)]


def test_rerank_texts_uses_flashrank_order(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    first = MagicMock(id=1, score=0.9)
    second = MagicMock(id=0, score=0.1)
    ranker = MagicMock()
    ranker.rerank.return_value = [first, second]
    with patch("services.rerank._get_ranker", return_value=ranker):
        ordered = rerank_texts("ELLA", ["unrelated", "ELLA transfers"])
    assert [index for index, _score in ordered] == [1, 0]
