from eval.graders import (
    abstain,
    citations,
    must_include_facts,
    outcome,
    retrieval_ids,
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


def test_tool_used_any_of_is_diagnostic_by_default():
    grade = tool_used(["list_papers", "read_paper"], ["search_library", "read_paper"])
    assert grade.passed is True
    assert grade.required is False
    missed = tool_used(["todo_write"], ["search_library", "read_paper"])
    assert missed.passed is False
    assert missed.required is False


def test_tool_used_none_of():
    grade = tool_used(["search_library"], none_of=["web_search"])
    assert grade.passed is True
    banned = tool_used(["search_library", "web_search"], none_of=["web_search"])
    assert banned.passed is False


def test_must_include_facts():
    ok = must_include_facts(
        "ZXQELLA7 transfers sparse models across tasks.",
        ["sparse", "transfer"],
    )
    assert ok.passed is True
    missing = must_include_facts("ZXQELLA7 is a paper.", ["sparse", "transfer"])
    assert missing.passed is False
    assert "sparse" in missing.detail


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


def test_retrieval_ids_set_difference():
    ok = retrieval_ids(
        ["aaa"],
        must_include=["aaa"],
        must_exclude=["bbb"],
    )
    assert ok.passed is True
    leaked = retrieval_ids(["aaa", "bbb"], must_exclude=["bbb"])
    assert leaked.passed is False


def test_outcome_excludes():
    ok = outcome("no matching paper", excludes=["ZXQ-HT-2023-014"])
    assert ok.passed is True
    leaked = outcome("see ZXQ-HT-2023-014", excludes=["ZXQ-HT-2023-014"])
    assert leaked.passed is False
    grade = outcome(
        "The library does not cover SOTA.", contains_any=["does not", "无法"]
    )
    assert grade.passed is True
    diagnostic = tool_used(["search_library"], ["search_library"])
    score, passed = score_task([grade, diagnostic])
    assert passed is True
    assert score == 1.0


def test_diagnostic_failure_does_not_fail_the_task():
    required = must_include_facts("sparse transfer", ["sparse"])
    diagnostic = tool_used(["todo_write"], ["search_library"], required=False)
    score, passed = score_task([required, diagnostic])
    assert passed is True
    assert diagnostic.passed is False
    assert score == 0.5
