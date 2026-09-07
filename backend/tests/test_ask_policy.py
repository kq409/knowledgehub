import uuid

from schemas import QueryKind
from services.ask import format_evidence
from services.ask_policy import (
    EvidenceQuality,
    abstain_answer,
    classify_query,
    decide_ask,
)
from services.retrieval import LibraryPaper, RetrievalHit

PAPER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SOTA_QUESTION = "What is the SOTA robot learning algorithm so far in 2026?"
LIBRARY_QUESTION = "What does ELLA claim about transfer?"


def _hit(
    *,
    similarity: float = 0.68,
    title: str = "ELLA",
    year: int | None = 2006,
    source_id: uuid.UUID = PAPER_ID,
    text: str = "ELLA is an efficient lifelong learning algorithm.",
) -> RetrievalHit:
    return RetrievalHit(
        source_type="paper",
        source_id=source_id,
        chunk_id=uuid.uuid4(),
        title=title,
        page=1,
        section="Abstract",
        text=text,
        similarity=similarity,
        year=year,
    )


def test_classify_sota_question_as_field_wide():
    assert classify_query(SOTA_QUESTION) is QueryKind.field_wide
    assert classify_query("目前最好的机器人学习算法是什么？") is QueryKind.field_wide
    assert (
        classify_query("What is the state-of-the-art method?") is QueryKind.field_wide
    )


def test_classify_ordinary_library_question():
    assert classify_query(LIBRARY_QUESTION) is QueryKind.library
    assert classify_query("Summarize the ELLA experiments.") is QueryKind.library


def test_ella_sota_question_skips_llm_and_suggests_external_search():
    papers = [LibraryPaper(title="ELLA", year=2006)]
    hits = [_hit(), _hit(similarity=0.61)]
    decision = decide_ask(
        SOTA_QUESTION, hits, papers, min_similarity=0.35, now_year=2026
    )

    assert decision.query_kind is QueryKind.field_wide
    assert decision.quality is EvidenceQuality.unsupported
    assert decision.skip_llm is True
    assert decision.suggest_external_search is True
    assert decision.insufficient_evidence is True
    assert decision.generation_hits == ()
    assert len(decision.citation_hits) == 2
    assert decision.coverage.paper_count == 1
    assert decision.coverage.unique_retrieved_papers == 1
    assert decision.coverage.years == [2006]

    answer = abstain_answer(SOTA_QUESTION, decision)
    assert "field-wide" in answer.lower() or "SOTA" in answer
    assert "ELLA" in answer
    assert "2006" in answer
    assert "search outside" in answer.lower()


def test_ordinary_library_question_still_generates():
    papers = [LibraryPaper(title="ELLA", year=2006)]
    hits = [_hit(similarity=0.68)]
    decision = decide_ask(
        LIBRARY_QUESTION, hits, papers, min_similarity=0.35, now_year=2026
    )

    assert decision.query_kind is QueryKind.library
    assert decision.skip_llm is False
    assert decision.suggest_external_search is False
    assert decision.quality is EvidenceQuality.ok
    assert decision.insufficient_evidence is False
    assert len(decision.generation_hits) == 1


def test_weak_ordinary_question_is_cautious_not_abstained():
    papers = [LibraryPaper(title="ELLA", year=2006)]
    hits = [_hit(similarity=0.20)]
    decision = decide_ask(
        LIBRARY_QUESTION, hits, papers, min_similarity=0.35, now_year=2026
    )

    assert decision.query_kind is QueryKind.library
    assert decision.quality is EvidenceQuality.low
    assert decision.skip_llm is False
    assert decision.insufficient_evidence is True
    assert decision.suggest_external_search is False
    assert len(decision.generation_hits) == 1


def test_empty_hits_refuse_without_citations_or_external_search():
    papers = [LibraryPaper(title="ELLA", year=2006)]
    decision = decide_ask(
        LIBRARY_QUESTION, [], papers, min_similarity=0.35, now_year=2026
    )

    assert decision.skip_llm is True
    assert decision.suggest_external_search is False
    assert decision.citation_hits == ()
    assert decision.generation_hits == ()
    answer = abstain_answer(LIBRARY_QUESTION, decision)
    assert "does not support" in answer.lower()
    assert "SOTA" not in answer


def test_empty_hits_field_wide_still_suggests_external_search():
    decision = decide_ask(SOTA_QUESTION, [], [], min_similarity=0.35, now_year=2026)
    assert decision.suggest_external_search is True
    assert decision.skip_llm is True
    assert decision.citation_hits == ()


def test_field_wide_with_recent_multi_paper_coverage_can_generate():
    papers = [
        LibraryPaper(title="A", year=2024),
        LibraryPaper(title="B", year=2025),
        LibraryPaper(title="C", year=2026),
    ]
    hits = [
        _hit(title="A", year=2024, source_id=uuid.uuid4(), similarity=0.8),
        _hit(title="B", year=2025, source_id=uuid.uuid4(), similarity=0.77),
        _hit(title="C", year=2026, source_id=uuid.uuid4(), similarity=0.74),
    ]
    decision = decide_ask(
        SOTA_QUESTION, hits, papers, min_similarity=0.35, now_year=2026
    )
    assert decision.skip_llm is False
    assert decision.quality is EvidenceQuality.ok
    assert decision.suggest_external_search is False


def test_field_wide_old_library_unsupported_even_with_several_papers():
    papers = [
        LibraryPaper(title="A", year=2006),
        LibraryPaper(title="B", year=2008),
        LibraryPaper(title="C", year=2010),
    ]
    hits = [
        _hit(title="A", year=2006, source_id=uuid.uuid4(), similarity=0.7),
        _hit(title="B", year=2008, source_id=uuid.uuid4(), similarity=0.66),
        _hit(title="C", year=2010, source_id=uuid.uuid4(), similarity=0.64),
    ]
    decision = decide_ask(
        SOTA_QUESTION, hits, papers, min_similarity=0.35, now_year=2026
    )
    assert decision.skip_llm is True
    assert decision.suggest_external_search is True


def test_chinese_field_wide_abstain_answer():
    papers = [LibraryPaper(title="ELLA", year=2006)]
    hits = [_hit()]
    decision = decide_ask(
        "2026年最先进的机器人学习算法是什么？",
        hits,
        papers,
        min_similarity=0.35,
        now_year=2026,
    )
    answer = abstain_answer("2026年最先进的机器人学习算法是什么？", decision)
    assert "文献库" in answer
    assert "库外搜索" in answer


def test_format_evidence_includes_year_and_inventory():
    papers = [LibraryPaper(title="ELLA", year=2006)]
    hits = [_hit()]
    text = format_evidence(hits, papers)
    assert "Library: 1 paper(s) (ELLA (2006))" in text
    assert "[1] Paper — ELLA (2006, p. 1, Abstract)" in text
    assert "Years: 2006" in text
