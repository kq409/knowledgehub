import uuid

from models import Note, NoteSourceType, ProcessingStatus
from schemas import ConnectReasonMode
from services.connect import (
    PaperCandidate,
    aggregate_paper_hits,
    apply_reasons,
    papers_to_auto_link,
    parse_reason_payload,
    query_text_for_note,
    select_candidates,
)
from services.retrieval import RetrievalHit


def _hit(paper_id: uuid.UUID, similarity: float, title: str = "Paper") -> RetrievalHit:
    return RetrievalHit(
        source_type="paper",
        source_id=paper_id,
        chunk_id=uuid.uuid4(),
        title=title,
        page=1,
        section="Abstract",
        text="chunk about hybrid retrieval",
        similarity=similarity,
        year=2020,
    )


def _candidate(
    paper_id: uuid.UUID,
    similarity: float,
    title: str = "Paper",
) -> PaperCandidate:
    return PaperCandidate(
        paper_id=paper_id,
        title=title,
        year=2020,
        similarity=similarity,
        snippet="chunk about hybrid retrieval",
        page=1,
        section="Abstract",
        chunk_id=uuid.uuid4(),
    )


def test_query_text_for_voice_note_uses_structured_fields():
    note = Note(
        source_type=NoteSourceType.voice.value,
        title="Hybrid idea",
        summary="Try dense plus BM25",
        observations=["Dense misses terms"],
        hypotheses=[],
        questions=[],
        next_steps=[],
        tags=[],
        cleaned_transcript="I should try hybrid retrieval.",
        processing_status=ProcessingStatus.ready.value,
    )
    text = query_text_for_note(note)
    assert "Hybrid idea" in text
    assert "Try dense plus BM25" in text
    assert "Dense misses terms" in text
    assert "hybrid retrieval" in text


def test_query_text_for_handwritten_uses_extracted_text():
    note = Note(
        source_type=NoteSourceType.handwritten.value,
        title="Scan",
        extracted_text="Margin note about ELLA",
        processing_status=ProcessingStatus.ready.value,
    )
    assert query_text_for_note(note) == "Margin note about ELLA"


def test_aggregate_paper_hits_keeps_max_similarity_chunk():
    paper_a = uuid.uuid4()
    paper_b = uuid.uuid4()
    hits = [
        _hit(paper_a, 0.40, "A"),
        _hit(paper_a, 0.81, "A"),
        _hit(paper_b, 0.70, "B"),
        RetrievalHit(
            source_type="voice",
            source_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            title="Note",
            page=None,
            section=None,
            text="ignore me",
            similarity=0.99,
        ),
    ]
    ranked = aggregate_paper_hits(hits)
    assert [item.title for item in ranked] == ["A", "B"]
    assert ranked[0].similarity == 0.81


def test_select_candidates_drops_low_scores_and_skips():
    keep = uuid.uuid4()
    skipped = uuid.uuid4()
    low = uuid.uuid4()
    ranked = [
        _candidate(keep, 0.9, "Keep"),
        _candidate(skipped, 0.8, "Skipped"),
        _candidate(low, 0.2, "Low"),
    ]
    selected = select_candidates(
        ranked,
        min_similarity=0.35,
        skipped={skipped},
        top_k=5,
    )
    assert [item.paper_id for item in selected] == [keep]


def test_select_candidates_respects_top_k():
    ranked = [_candidate(uuid.uuid4(), 0.9 - i * 0.01) for i in range(6)]
    selected = select_candidates(ranked, min_similarity=0.35, skipped=set(), top_k=3)
    assert len(selected) == 3


def test_papers_to_auto_link_skips_already_linked_and_cap():
    linked = uuid.uuid4()
    a = uuid.uuid4()
    b = uuid.uuid4()
    candidates = [
        _candidate(linked, 0.9, "Linked"),
        _candidate(a, 0.8, "A"),
        _candidate(b, 0.7, "B"),
    ]
    to_link = papers_to_auto_link(
        candidates, already_linked={linked}, remaining_slots=1
    )
    assert [item.paper_id for item in to_link] == [a]


def test_papers_to_auto_link_empty_when_cap_full():
    candidates = [_candidate(uuid.uuid4(), 0.9)]
    assert (
        papers_to_auto_link(candidates, already_linked=set(), remaining_slots=0) == []
    )


def test_apply_reasons_falls_back_to_snippet():
    paper_id = uuid.uuid4()
    item = _candidate(paper_id, 0.7)
    explained = apply_reasons(
        [item],
        {},
        reason_mode=ConnectReasonMode.llm,
    )
    assert explained[0][1] == item.snippet


def test_apply_reasons_uses_llm_when_present():
    paper_id = uuid.uuid4()
    item = _candidate(paper_id, 0.7)
    explained = apply_reasons(
        [item],
        {paper_id: "Shares the hybrid retrieval hypothesis."},
        reason_mode=ConnectReasonMode.llm,
    )
    assert explained[0][1] == "Shares the hybrid retrieval hypothesis."


def test_parse_reason_payload_skips_bad_ids():
    paper_id = uuid.uuid4()
    parsed = parse_reason_payload(
        {
            "reasons": [
                {"paper_id": str(paper_id), "reason": "Matches the note."},
                {"paper_id": "not-a-uuid", "reason": "nope"},
                {"paper_id": str(uuid.uuid4()), "reason": "  "},
            ]
        }
    )
    assert parsed == {paper_id: "Matches the note."}
