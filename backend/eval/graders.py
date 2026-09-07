"""Deterministic graders over Ask/Chat transcripts. LLM faithfulness is opt-in."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from services.agent.gate import check_deterministic, extract_citation_indices


@dataclass(frozen=True)
class Grade:
    name: str
    passed: bool
    score: float
    detail: str
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float(passed: bool) -> float:
    return 1.0 if passed else 0.0


def retrieval_titles(
    retrieved_titles: list[str],
    *,
    must_include: list[str] | None = None,
    must_exclude: list[str] | None = None,
    k: int = 8,
) -> Grade:
    top = retrieved_titles[:k]
    missing = [title for title in (must_include or []) if title not in top]
    present = [title for title in (must_exclude or []) if title in top]
    passed = not missing and not present
    bits: list[str] = []
    if missing:
        bits.append("missing " + ", ".join(missing))
    if present:
        bits.append("unexpected " + ", ".join(present))
    if not bits:
        bits.append(f"top-{k} titles acceptable")
    return Grade(
        name="retrieval_titles",
        passed=passed,
        score=_as_float(passed),
        detail="; ".join(bits),
    )


def abstain(
    *,
    insufficient_evidence: bool,
    suggest_external_search: bool,
    expect: str,
) -> Grade:
    expected = expect.strip().lower()
    if expected == "insufficient_evidence":
        passed = insufficient_evidence
        detail = (
            "abstained as expected"
            if passed
            else "expected insufficient_evidence, got an answerable library hit"
        )
    elif expected == "suggest_external_search":
        passed = suggest_external_search
        detail = (
            "suggested external search"
            if passed
            else "expected suggest_external_search"
        )
    elif expected == "none":
        passed = not insufficient_evidence
        detail = (
            "library evidence was enough"
            if passed
            else "expected a grounded library answer, got an abstain"
        )
    else:
        passed = False
        detail = f"unknown abstain expect {expect!r}"
    return Grade(
        name="abstain",
        passed=passed,
        score=_as_float(passed),
        detail=detail,
    )


def citations(
    answer: str,
    registered_indices: set[int],
    *,
    require_indices: bool = False,
    tool_calls_made: int = 1,
) -> Grade:
    cited = extract_citation_indices(answer)
    problems = check_deterministic(answer, registered_indices, tool_calls_made)
    hard = [problem for problem in problems if problem.hard]
    # Ask has no tools; skip the "wrote without a tool call" check.
    hard = [
        problem
        for problem in hard
        if problem.kind.value != "no_evidence_gathered" or tool_calls_made > 0
    ]
    if require_indices and not cited:
        return Grade(
            name="citations",
            passed=False,
            score=0.0,
            detail="answer cites no [n] indices",
        )
    passed = not hard
    detail = (
        "citation numbers match the registry"
        if passed
        else "; ".join(problem.detail for problem in hard)
    )
    return Grade(
        name="citations",
        passed=passed,
        score=_as_float(passed),
        detail=detail,
    )


def tool_used(tools_called: list[str], any_of: list[str]) -> Grade:
    used = set(tools_called)
    wanted = set(any_of)
    hit = sorted(used & wanted)
    passed = bool(hit)
    return Grade(
        name="tool_used",
        passed=passed,
        score=_as_float(passed),
        detail=(
            "used " + ", ".join(hit)
            if passed
            else "did not use any of " + ", ".join(any_of)
        ),
    )


def transcript(
    *,
    tools_called: list[str],
    answer: str,
    registered_indices: set[int],
    max_tool_calls: int | None = None,
) -> Grade:
    problems: list[str] = []
    if max_tool_calls is not None and len(tools_called) > max_tool_calls:
        problems.append(f"{len(tools_called)} tool calls exceeds max {max_tool_calls}")
    cited = extract_citation_indices(answer)
    fabricated = sorted(cited - registered_indices)
    if fabricated:
        problems.append(
            "fabricated citations " + ", ".join(f"[{n}]" for n in fabricated)
        )
    passed = not problems
    return Grade(
        name="transcript",
        passed=passed,
        score=_as_float(passed),
        detail="; ".join(problems) if problems else "transcript constraints held",
    )


def outcome(
    answer: str,
    *,
    contains: list[str] | None = None,
    contains_any: list[str] | None = None,
    gate_status: str | None = None,
    expect_gate: str | None = None,
) -> Grade:
    text = answer.lower()
    missing = [needle for needle in (contains or []) if needle.lower() not in text]
    any_needles = contains_any or []
    any_hit = not any_needles or any(needle.lower() in text for needle in any_needles)
    gate_ok = expect_gate is None or (gate_status or "") == expect_gate
    passed = not missing and any_hit and gate_ok
    bits: list[str] = []
    if missing:
        bits.append("missing " + ", ".join(missing))
    if any_needles and not any_hit:
        bits.append("none of " + ", ".join(any_needles))
    if not gate_ok:
        bits.append(f"gate {gate_status!r} != {expect_gate!r}")
    if not bits:
        bits.append("outcome matched")
    return Grade(
        name="outcome",
        passed=passed,
        score=_as_float(passed),
        detail="; ".join(bits),
    )


def score_task(grades: list[Grade]) -> tuple[float, bool]:
    if not grades:
        return 0.0, False
    total = sum(grade.score for grade in grades) / len(grades)
    required_ok = all(grade.passed for grade in grades if grade.required)
    return total, required_ok


@dataclass
class FaithfulnessInput:
    question: str
    answer: str
    citations: list[Any] = field(default_factory=list)
    tool_calls_made: int = 0


async def faithfulness(payload: FaithfulnessInput, gate: Any) -> Grade:
    """Reuse EvidenceGate. Unknown/unchecked does not fail the task."""
    verdict = await gate.check(
        question=payload.question,
        answer=payload.answer,
        citations=payload.citations,
        tool_calls_made=payload.tool_calls_made,
    )
    status = getattr(verdict.status, "value", str(verdict.status))
    if status in {"unchecked", "supported"}:
        return Grade(
            name="faithfulness",
            passed=True,
            score=1.0 if status == "supported" else 0.5,
            detail=f"judge {status}: {verdict.reason or 'ok'}",
            required=False,
        )
    return Grade(
        name="faithfulness",
        passed=False,
        score=0.0,
        detail=f"judge {status}: {verdict.reason}",
        required=False,
    )
