import json
import uuid
from unittest.mock import MagicMock

import pytest

from schemas import ChatCitation, CitationSourceType, GateProblemKind, GateStatus
from services.agent.gate import (
    EvidenceGate,
    GateMode,
    _problems_from_payload,
    _short_reason,
    check_deterministic,
    compact_answer_citations,
    extract_citation_indices,
    gate_mode_from_env,
    is_local_endpoint,
)

MODEL = "test-model"


def citation(index: int, snippet: str = "ELLA transfers across tasks.") -> ChatCitation:
    return ChatCitation(
        index=index,
        source_type=CitationSourceType.paper,
        source_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        title="ELLA",
        snippet=snippet,
    )


def judge_llm(*replies: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=item))])
        for item in replies
    ]
    return client


def cloud_gate(client: MagicMock) -> EvidenceGate:
    return EvidenceGate(
        llm_client=client,
        llm_model=MODEL,
        base_url="https://api.deepseek.com/v1",
        mode=GateMode.auto,
    )


def test_citation_numbers_come_out_of_the_prose():
    assert extract_citation_indices("Backed by [1] and [12].") == {1, 12}


def test_grouped_citations_count_individually():
    assert extract_citation_indices("Both agree [2, 5] here.") == {2, 5}


def test_markdown_links_are_not_citations():
    assert extract_citation_indices("See [the paper](http://x) for more.") == set()
    assert extract_citation_indices("See [1](http://x) for more.") == set()


def test_compact_drops_unread_footnotes_and_renumbers_from_one():
    answer, citations = compact_answer_citations(
        "ELLA is a lifelong learner [3].",
        [citation(1), citation(2), citation(3, "shared basis"), citation(6)],
    )
    assert answer == "ELLA is a lifelong learner [1]."
    assert [item.index for item in citations] == [1]
    assert citations[0].snippet == "shared basis"


def test_compact_keeps_first_appearance_order():
    answer, citations = compact_answer_citations(
        "Later work [5] builds on ELLA [3].",
        [citation(3, "ella"), citation(5, "later")],
    )
    assert answer == "Later work [1] builds on ELLA [2]."
    assert [item.snippet for item in citations] == ["later", "ella"]


def test_compact_rewrites_grouped_citations():
    answer, citations = compact_answer_citations(
        "Both agree [3, 5].",
        [citation(3, "a"), citation(5, "b"), citation(9)],
    )
    assert answer == "Both agree [1, 2]."
    assert [item.snippet for item in citations] == ["a", "b"]


def test_compact_leaves_an_uncited_answer_without_footnotes():
    answer, citations = compact_answer_citations(
        "No numbers here.", [citation(1), citation(2)]
    )
    assert answer == "No numbers here."
    assert citations == []


def test_a_number_no_tool_handed_out_is_a_hard_failure():
    problems = check_deterministic("ELLA is 2026 SOTA [1].", set(), tool_calls_made=1)

    kinds = [problem.kind for problem in problems]
    assert GateProblemKind.fabricated_citation in kinds
    assert all(problem.hard for problem in problems)
    assert "[1]" in problems[0].detail


def test_zero_tool_answer_without_citations_is_allowed():
    assert (
        check_deterministic("Hello — how can I help?", set(), tool_calls_made=0) == []
    )


def test_evidence_gathered_but_never_cited_is_only_a_warning():
    problems = check_deterministic("Probably yes.", {1, 2}, tool_calls_made=2)

    assert [problem.kind for problem in problems] == [GateProblemKind.uncited_answer]
    assert not problems[0].hard


def test_a_properly_cited_answer_raises_nothing():
    assert check_deterministic("ELLA transfers [1].", {1, 2}, tool_calls_made=2) == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:1234/v1",
        "http://0.0.0.0:8000",
        "http://ollama:11434/v1",
        "http://host.docker.internal:11434/v1",
        "http://my-box.local:11434/v1",
        "http://192.168.1.40:11434/v1",
        "http://[::1]:11434/v1",
        "",
        None,
    ],
)
def test_endpoints_on_this_machine_or_network_are_local(base_url):
    assert is_local_endpoint(base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.deepseek.com/v1",
        "https://api.openai.com/v1",
        "https://openrouter.ai/api/v1",
        "api.deepseek.com/v1",
    ],
)
def test_endpoints_off_the_machine_are_not_local(base_url):
    assert not is_local_endpoint(base_url)


def test_gate_mode_falls_back_to_auto_on_nonsense():
    assert gate_mode_from_env("sideways") is GateMode.auto
    assert gate_mode_from_env("") is GateMode.auto
    assert gate_mode_from_env("LLM") is GateMode.llm
    assert gate_mode_from_env("deterministic") is GateMode.deterministic


def test_a_local_model_gets_no_reviewer():
    gate = EvidenceGate(
        llm_client=MagicMock(),
        llm_model=MODEL,
        base_url="http://ollama:11434/v1",
        mode=GateMode.auto,
    )

    assert not gate.uses_judge()


def test_a_cloud_model_gets_a_reviewer():
    assert cloud_gate(MagicMock()).uses_judge()


def test_deterministic_mode_overrides_a_cloud_endpoint():
    gate = EvidenceGate(
        llm_client=MagicMock(),
        llm_model=MODEL,
        base_url="https://api.deepseek.com/v1",
        mode=GateMode.deterministic,
    )

    assert not gate.uses_judge()


def test_llm_mode_overrides_a_local_endpoint():
    gate = EvidenceGate(
        llm_client=MagicMock(),
        llm_model=MODEL,
        base_url="http://ollama:11434/v1",
        mode=GateMode.llm,
    )

    assert gate.uses_judge()


def test_the_reviewer_uses_its_own_model_when_one_is_named():
    client = judge_llm('{"verdict": "supported", "reason": "ok"}')
    gate = EvidenceGate(
        llm_client=client,
        llm_model=MODEL,
        base_url="https://api.deepseek.com/v1",
        judge_model="cheap-model",
    )

    assert gate.judge_model == "cheap-model"


async def test_reviewer_clears_an_answer_it_believes():
    gate = cloud_gate(judge_llm('{"verdict": "supported", "unsupported_claims": []}'))

    verdict = await gate.check(
        question="Does ELLA transfer?",
        answer="ELLA transfers across tasks [1].",
        citations=[citation(1)],
        tool_calls_made=1,
    )

    assert verdict.status is GateStatus.supported
    assert verdict.ok
    assert "llm reviewer" in verdict.checked_by


async def test_reviewer_rejection_names_the_claim():
    gate = cloud_gate(
        judge_llm(
            '{"verdict": "unsupported", "reason": "The snippet says nothing about '
            'benchmarks.", "unsupported_claims": ["ELLA beats every baseline"]}'
        )
    )

    verdict = await gate.check(
        question="Is ELLA the best?",
        answer="ELLA beats every baseline [1].",
        citations=[citation(1)],
        tool_calls_made=1,
    )

    assert verdict.status is GateStatus.unsupported
    assert not verdict.ok
    assert "ELLA beats every baseline" in verdict.reason
    assert "benchmarks" in verdict.feedback()


async def test_reviewer_gets_one_retry_on_unparseable_json():
    client = judge_llm(
        "Sure! Here you go.",
        '{"verdict": "supported", "unsupported_claims": []}',
    )
    gate = cloud_gate(client)

    verdict = await gate.check(
        question="Does ELLA transfer?",
        answer="ELLA transfers [1].",
        citations=[citation(1)],
        tool_calls_made=1,
    )

    assert client.chat.completions.create.call_count == 2
    assert verdict.status is GateStatus.supported


async def test_a_reviewer_that_never_returns_json_fails_open():
    gate = cloud_gate(judge_llm("thinking...", "still thinking..."))

    verdict = await gate.check(
        question="Does ELLA transfer?",
        answer="ELLA transfers [1].",
        citations=[citation(1)],
        tool_calls_made=1,
    )

    assert verdict.status is GateStatus.unchecked
    assert verdict.ok
    assert [problem.kind for problem in verdict.problems] == [
        GateProblemKind.judge_unavailable
    ]


async def test_an_unreachable_reviewer_fails_open():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("connection refused")
    gate = cloud_gate(client)

    verdict = await gate.check(
        question="Does ELLA transfer?",
        answer="ELLA transfers [1].",
        citations=[citation(1)],
        tool_calls_made=1,
    )

    assert verdict.status is GateStatus.unchecked
    assert verdict.ok


async def test_a_fabricated_number_never_reaches_the_reviewer():
    client = judge_llm('{"verdict": "supported", "unsupported_claims": []}')
    gate = cloud_gate(client)

    verdict = await gate.check(
        question="Is ELLA the best?",
        answer="ELLA is 2026 SOTA [1].",
        citations=[],
        tool_calls_made=1,
    )

    client.chat.completions.create.assert_not_called()
    assert verdict.status is GateStatus.unsupported
    assert "[1]" in verdict.feedback()


async def test_without_a_client_only_the_cheap_checks_run():
    gate = EvidenceGate()

    verdict = await gate.check(
        question="Anything?",
        answer="ELLA transfers [1].",
        citations=[citation(1)],
        tool_calls_made=1,
    )

    assert verdict.status is GateStatus.supported
    assert verdict.checked_by == "deterministic"


async def test_zero_tool_uncited_answer_skips_the_reviewer():
    client = judge_llm('{"verdict": "unsupported", "reason": "should not run"}')
    gate = cloud_gate(client)

    verdict = await gate.check(
        question="Hello",
        answer="Hello — what would you like to explore?",
        citations=[],
        tool_calls_made=0,
    )

    client.chat.completions.create.assert_not_called()
    assert verdict.status is GateStatus.supported
    assert verdict.ok


def test_gate_feedback_is_an_internal_check():
    from services.agent.gate import GateProblem, GateVerdict

    verdict = GateVerdict(
        status=GateStatus.unsupported,
        problems=[
            GateProblem(
                kind=GateProblemKind.fabricated_citation,
                detail="The answer cites [99], but no tool ever returned that evidence.",
            )
        ],
    )
    text = verdict.feedback()
    assert "not the researcher" in text
    assert "Do not thank the user" in text
    assert "Fix this before answering" not in text
    assert "[99]" in text


async def test_inventory_snippets_reach_the_reviewer():
    client = judge_llm(
        '{"verdict": "supported", "reason": "counts match the inventory.", '
        '"unsupported_claims": []}'
    )
    gate = cloud_gate(client)
    papers = ChatCitation(
        index=1,
        source_type=CitationSourceType.paper,
        source_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        title="Paper library",
        section="inventory",
        snippet="Library: 1 paper(s), 1 ready to search. Publication years: 2006.",
    )
    empty_docs = ChatCitation(
        index=2,
        source_type=CitationSourceType.document,
        source_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        title="Document library",
        section="inventory",
        snippet="The document library is empty.",
    )

    verdict = await gate.check(
        question="give me an overview on my Library",
        answer="The library holds 1 paper from 2006. The document library is empty.",
        citations=[papers, empty_docs],
        tool_calls_made=2,
    )

    payload = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Publication years: 2006" in payload
    assert "The document library is empty." in payload
    assert verdict.status is GateStatus.supported


def test_unsupported_without_quoted_claims_is_treated_as_supported():
    problems = _problems_from_payload(
        {
            "verdict": "unsupported",
            "reason": (
                "The answer claims ELLA is from 1998, but the snippets show "
                "ELLA is 1998. So all claims are supported. I will return supported."
            ),
            "unsupported_claims": [],
        }
    )
    assert problems == []


def test_unsupported_reason_is_clipped_to_one_short_sentence():
    ramble = (
        "The answer claims ELLA is from 1998 and Progress & Compress is from 2016, "
        "but the snippets show ELLA is 1998. However, everything matches. "
        + ("x" * 400)
    )
    problems = _problems_from_payload(
        {
            "verdict": "unsupported",
            "reason": ramble,
            "unsupported_claims": ["All four are in continual/lifelong learning"],
        }
    )
    assert len(problems) == 1
    assert problems[0].hard
    assert "Unsupported:" in problems[0].detail
    assert len(problems[0].detail) < len(ramble)
    assert "continual/lifelong learning" in problems[0].detail


def test_short_reason_keeps_a_single_sentence():
    assert _short_reason("Counts match. Extra recap.") == "Counts match."


async def test_rambling_unsupported_verdict_does_not_block():
    gate = cloud_gate(
        judge_llm(
            json.dumps(
                {
                    "verdict": "unsupported",
                    "reason": (
                        "The answer claims ELLA is from 1998, but the snippets "
                        "show ELLA is 1998. I see no unsupported claims."
                    ),
                    "unsupported_claims": [],
                }
            )
        )
    )
    verdict = await gate.check(
        question="give me an overview on my Library",
        answer="ELLA (1998). The library holds 1 paper.",
        citations=[citation(1, snippet="ELLA | 1998")],
        tool_calls_made=1,
    )
    assert verdict.status is GateStatus.supported
    assert verdict.ok


def test_unaddressed_parts_are_a_hard_incomplete():
    problems = _problems_from_payload(
        {
            "verdict": "supported",
            "reason": "Claims hold, but the second half is missing.",
            "unsupported_claims": [],
            "unaddressed_parts": ["what the library still lacks"],
            "impossible": False,
        }
    )
    assert [item.kind for item in problems] == [GateProblemKind.unaddressed_part]
    assert all(item.hard for item in problems)


def test_impossible_stops_retries_and_ignores_leftover_parts():
    problems = _problems_from_payload(
        {
            "verdict": "supported",
            "reason": "No paper in the library covers this.",
            "unsupported_claims": [],
            "unaddressed_parts": ["a comparison table"],
            "impossible": True,
        }
    )
    kinds = [item.kind for item in problems]
    assert GateProblemKind.impossible in kinds
    assert GateProblemKind.unaddressed_part not in kinds


def test_supported_with_no_leftover_is_still_empty():
    assert (
        _problems_from_payload(
            {
                "verdict": "supported",
                "reason": "ok",
                "unsupported_claims": [],
            }
        )
        == []
    )
