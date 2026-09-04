import uuid

from services.retrieval import (
    DEFAULT_MIN_SIMILARITY,
    RetrievalHit,
    filter_by_min_similarity,
    is_insufficient,
)


def _hit(similarity: float) -> RetrievalHit:
    return RetrievalHit(
        source_type="paper",
        source_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        title="ELLA",
        page=1,
        section="Abstract",
        text="chunk",
        similarity=similarity,
        year=2006,
    )


def test_filter_by_min_similarity_drops_low_scores():
    hits = [_hit(0.68), _hit(0.20), _hit(0.35)]
    kept = filter_by_min_similarity(hits, DEFAULT_MIN_SIMILARITY)
    assert [hit.similarity for hit in kept] == [0.68, 0.35]


def test_is_insufficient_uses_max_similarity():
    assert is_insufficient([], 0.35) is True
    assert is_insufficient([_hit(0.20)], 0.35) is True
    assert is_insufficient([_hit(0.68), _hit(0.20)], 0.35) is False
