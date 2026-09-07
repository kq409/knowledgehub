import uuid

from services.retrieval import RetrievalHit, _apply_rerank


def _hit(text: str) -> RetrievalHit:
    return RetrievalHit(
        source_type="paper",
        source_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        title=text,
        page=1,
        section="Abstract",
        text=text,
        similarity=0.5,
    )


def test_apply_rerank_uses_flashrank_order(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setattr("services.rerank.rerank_enabled", lambda: True)
    monkeypatch.setattr(
        "services.rerank.rerank_texts",
        lambda _query, texts: [(1, 0.9), (0, 0.1)],
    )
    first = _hit("first")
    second = _hit("second")
    reranked = _apply_rerank("query", [first, second], keep=2)
    assert [hit.text for hit in reranked] == ["second", "first"]
    assert reranked[0].rank_score == 0.9


def test_apply_rerank_is_identity_when_disabled(monkeypatch):
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    monkeypatch.setattr("services.rerank.rerank_enabled", lambda: False)
    first = _hit("first")
    second = _hit("second")
    reranked = _apply_rerank("query", [first, second], keep=2)
    assert [hit.text for hit in reranked] == ["first", "second"]
