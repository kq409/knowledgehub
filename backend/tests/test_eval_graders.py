from eval.graders import (
    abstain,
    citations,
    outcome,
    retrieval_titles,
    score_task,
    tool_used,
    transcript,
)


def test_retrieval_titles_requires_all_must_include():
    grade = retrieval_titles(
        ["ZXQELLA7: Efficient Lifelong Learning Algorithm", "Other"],
        must_include=["ZXQELLA7: Efficient Lifelong Learning Algorithm"],
        k=8,
    )
    assert grade.passed is True
    missing = retrieval_titles(
        ["Other"],
        must_include=["ZXQELLA7: Efficient Lifelong Learning Algorithm"],
        k=8,
    )
    assert missing.passed is False


def test_abstain_expect_insufficient():
    ok = abstain(
        insufficient_evidence=True,
        suggest_external_search=True,
        expect="insufficient_evidence",
    )
    assert ok.passed is True
    bad = abstain(
        insufficient_evidence=False,
        suggest_external_search=False,
        expect="insufficient_evidence",
    )
    assert bad.passed is False


def test_citations_drop_ask_no_tool_check():
    grade = citations(
        "ELLA transfers [1].", {1}, require_indices=True, tool_calls_made=0
    )
    assert grade.passed is True
    fabricated = citations("Made up [9].", {1}, tool_calls_made=1)
    assert fabricated.passed is False


def test_tool_used_any_of():
    grade = tool_used(["list_papers", "read_paper"], ["search_library", "read_paper"])
    assert grade.passed is True
    missed = tool_used(["todo_write"], ["search_library", "read_paper"])
    assert missed.passed is False


def test_transcript_flags_too_many_calls_and_fabricated_cites():
    too_many = transcript(
        tools_called=["search_library"] * 9,
        answer="ok [1]",
        registered_indices={1},
        max_tool_calls=8,
    )
    assert too_many.passed is False
    fake = transcript(
        tools_called=["search_library"],
        answer="see [3]",
        registered_indices={1},
    )
    assert fake.passed is False


def test_outcome_contains_any_and_score_task():
    grade = outcome(
        "The library does not cover SOTA.", contains_any=["does not", "无法"]
    )
    assert grade.passed is True
    score, passed = score_task(
        [grade, tool_used(["search_library"], ["search_library"])]
    )
    assert passed is True
    assert score == 1.0
