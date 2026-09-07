import uuid
from unittest.mock import MagicMock

import pytest

from schemas import ChatCitation, CitationSourceType, GateProblemKind, GateStatus
from services.agent.gate import (
    EvidenceGate,
    GateMode,
    check_deterministic,
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


def test_a_number_no_tool_handed_out_is_a_hard_failure():
    problems = check_deterministic("ELLA is 2026 SOTA [1].", set(), tool_calls_made=1)

    kinds = [problem.kind for problem in problems]
    assert GateProblemKind.fabricated_citation in kinds
    assert all(problem.hard for problem in problems)
    assert "[1]" in problems[0].detail


def test_answering_without_opening_anything_is_a_hard_failure():
    problems = check_deterministic("Transformers are great.", set(), tool_calls_made=0)

    assert [problem.kind for problem in problems] == [
        GateProblemKind.no_evidence_gathered
    ]


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
