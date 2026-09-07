import uuid

from services.retrieval import RetrievalHit, rrf_fuse


def _hit(title: str, similarity: float) -> RetrievalHit:
    return RetrievalHit(
        source_type="paper",
        source_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        title=title,
        page=1,
        section="Abstract",
        text=title,
        similarity=similarity,
    )


def test_rrf_prefers_items_high_in_both_lists():
    shared = _hit("both", 0.4)
    dense_only = _hit("dense", 0.9)
    lexical_only = _hit("lex", 0.0)
    fused = rrf_fuse(
        [
            [shared, dense_only],
            [shared, lexical_only],
        ]
    )
    assert fused[0].chunk_id == shared.chunk_id
    assert fused[0].rank_score is not None
    assert fused[0].rank_score > (fused[1].rank_score or 0)
    titles = [hit.title for hit in fused]
    assert titles[0] == "both"
    assert set(titles) == {"both", "dense", "lex"}


def test_rrf_keeps_best_dense_similarity_for_duplicate_chunk():
    chunk_id = uuid.uuid4()
    source_id = uuid.uuid4()
    low = RetrievalHit(
        source_type="paper",
        source_id=source_id,
        chunk_id=chunk_id,
        title="ELLA",
        page=1,
        section="Abstract",
        text="ELLA",
        similarity=0.1,
    )
    high = RetrievalHit(
        source_type="paper",
        source_id=source_id,
        chunk_id=chunk_id,
        title="ELLA",
        page=1,
        section="Abstract",
        text="ELLA",
        similarity=0.8,
    )
    fused = rrf_fuse([[low], [high]])
    assert len(fused) == 1
    assert fused[0].similarity == 0.8
