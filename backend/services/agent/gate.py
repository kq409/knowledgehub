"""The evidence gate: does the answer actually follow from what was read?

Two layers, deliberately split by what each is good at. Deterministic checks
always run and catch facts a model is bad at judging -- a citation number that
was never handed out, an answer written without opening anything. When the main
model runs in the cloud, a second LLM reads the answer against the snippets and
judges whether the prose is really supported.

The reviewer fails open. If it cannot be reached or cannot produce parseable
JSON, the answer still goes out, marked as unchecked. A flaky reviewer must
never take the chat down with it.
"""

import asyncio
import ipaddress
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI

from schemas import (
    ChatCitation,
    ChatGateProblem,
    ChatVerdict,
    GateProblemKind,
    GateStatus,
)

PROMPT_FILE = Path(__file__).resolve().parent.parent.parent / "judge_prompt.txt"
JUDGE_PROMPT = PROMPT_FILE.read_text().strip()

JUDGE_MAX_TOKENS = 2000
JUDGE_TEMPERATURE = 0.0
JUDGE_ATTEMPTS = 2
JUDGE_SNIPPET_CHARS = 700
MAX_JUDGE_CLAIMS = 3

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

LOCAL_HOSTNAMES = {
    "localhost",
    "ollama",
    "host.docker.internal",
    "host.containers.internal",
}
LOCAL_SUFFIXES = (".local", ".internal", ".localdomain")

CHECKED_BY_DETERMINISTIC = "deterministic"
CHECKED_BY_BOTH = "deterministic + llm reviewer"

JUDGE_UNAVAILABLE_DETAIL = (
    "The reviewer could not be reached, so nobody checked whether the wording "
    "matches the snippets."
)


class GateMode(str, Enum):
    """`auto` picks the reviewer based on where the main model lives."""

    auto = "auto"
    deterministic = "deterministic"
    llm = "llm"


@dataclass(frozen=True)
class GateProblem:
    kind: GateProblemKind
    detail: str
    hard: bool = True

    def as_schema(self) -> ChatGateProblem:
        return ChatGateProblem(kind=self.kind, detail=self.detail)


@dataclass(frozen=True)
class GateVerdict:
    status: GateStatus
    checked_by: str = CHECKED_BY_DETERMINISTIC
    problems: list[GateProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the answer may go out as-is. Soft warnings do not block."""
        return not any(problem.hard for problem in self.problems)

    @property
    def reason(self) -> str:
        return " ".join(problem.detail for problem in self.problems)

    def feedback(self) -> str:
        """What to tell the model so its next attempt is different."""
        issues = "\n".join(
            f"- {problem.detail}" for problem in self.problems if problem.hard
        )
        return (
            "Your draft answer did not pass the evidence check:\n"
            f"{issues}\n\n"
            "Fix this before answering again. Search or read what you need to "
            "back up the claim, or drop the claim and say plainly that the "
            "library does not cover it. Only cite numbers the tools gave you."
        )

    def as_schema(self, status: GateStatus | None = None) -> ChatVerdict:
        return ChatVerdict(
            status=status or self.status,
            reason=self.reason,
            checked_by=self.checked_by,
            problems=[problem.as_schema() for problem in self.problems],
        )


def extract_citation_indices(answer: str) -> set[int]:
    """Every [n] the answer claims, including grouped forms like [2, 5]."""
    found: set[int] = set()
    for group in CITATION_PATTERN.findall(answer):
        found.update(int(part) for part in group.split(","))
    return found


def is_local_endpoint(base_url: str | None) -> bool:
    """True when the model is served from this machine or this network.

    Unknown or unparseable endpoints count as local: the reviewer costs a real
    API call, so we only spend it when we are sure the traffic already leaves
    the machine anyway.
    """
    if not base_url or not base_url.strip():
        return True

    candidate = base_url.strip()
    if "//" not in candidate:
        candidate = f"//{candidate}"
    host = urlparse(candidate).hostname
    if not host:
        return True

    host = host.lower()
    if host in LOCAL_HOSTNAMES or host.endswith(LOCAL_SUFFIXES):
        return True
    if "." not in host and ":" not in host:
        return True

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_unspecified


def gate_mode_from_env(raw: str | None = None) -> GateMode:
    value = (raw if raw is not None else os.getenv("EVIDENCE_GATE_MODE") or "").strip()
    try:
        return GateMode(value.lower())
    except ValueError:
        return GateMode.auto


def check_deterministic(
    answer: str, registered: set[int], tool_calls_made: int
) -> list[GateProblem]:
    """Checks that need no model: cited numbers exist, something was read."""
    problems: list[GateProblem] = []
    cited = extract_citation_indices(answer)

    fabricated = sorted(cited - registered)
    if fabricated:
        numbers = ", ".join(f"[{index}]" for index in fabricated)
        available = (
            ", ".join(f"[{index}]" for index in sorted(registered))
            if registered
            else "none yet"
        )
        problems.append(
            GateProblem(
                kind=GateProblemKind.fabricated_citation,
                detail=(
                    f"The answer cites {numbers}, but no tool ever returned "
                    f"that evidence. Numbers actually registered: {available}."
                ),
            )
        )

    if tool_calls_made == 0:
        problems.append(
            GateProblem(
                kind=GateProblemKind.no_evidence_gathered,
                detail=(
                    "The answer was written without a single tool call, so "
                    "nothing in it comes from the library."
                ),
            )
        )

    if registered and not cited:
        problems.append(
            GateProblem(
                kind=GateProblemKind.uncited_answer,
                detail="Evidence was retrieved but the answer cites none of it.",
                hard=False,
            )
        )

    return problems


def format_evidence(citations: list[ChatCitation]) -> str:
    if not citations:
        return "(no evidence was registered)"
    lines: list[str] = []
    for citation in citations:
        where = citation.title
        if citation.page is not None:
            where = f"{where}, p. {citation.page}"
        snippet = citation.snippet.strip()[:JUDGE_SNIPPET_CHARS]
        lines.append(f"[{citation.index}] {where}\n{snippet}")
    return "\n\n".join(lines)


class EvidenceGate:
    """Runs the checks and decides whether the loop should try again."""

    def __init__(
        self,
        llm_client: OpenAI | None = None,
        llm_model: str | None = None,
        *,
        base_url: str | None = None,
        mode: GateMode | None = None,
        judge_model: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.base_url = base_url
        self.mode = mode or gate_mode_from_env()
        self.judge_model = judge_model or os.getenv("JUDGE_MODEL") or llm_model

    def uses_judge(self) -> bool:
        if self.llm_client is None or not self.judge_model:
            return False
        if self.mode is GateMode.deterministic:
            return False
        if self.mode is GateMode.llm:
            return True
        return not is_local_endpoint(self.base_url)

    async def check(
        self,
        *,
        question: str,
        answer: str,
        citations: list[ChatCitation],
        tool_calls_made: int,
    ) -> GateVerdict:
        registered = {citation.index for citation in citations}
        problems = check_deterministic(answer, registered, tool_calls_made)

        if not self.uses_judge():
            return GateVerdict(
                status=_status_for(problems),
                checked_by=CHECKED_BY_DETERMINISTIC,
                problems=problems,
            )

        if any(problem.hard for problem in problems):
            # The answer is going back to the model regardless; skip the call.
            return GateVerdict(
                status=GateStatus.unsupported,
                checked_by=CHECKED_BY_DETERMINISTIC,
                problems=problems,
            )

        judged = await self._judge(question, answer, citations)
        if judged is None:
            return GateVerdict(
                status=GateStatus.unchecked,
                checked_by=CHECKED_BY_DETERMINISTIC,
                problems=[
                    *problems,
                    GateProblem(
                        kind=GateProblemKind.judge_unavailable,
                        detail=JUDGE_UNAVAILABLE_DETAIL,
                        hard=False,
                    ),
                ],
            )

        problems = [*problems, *judged]
        return GateVerdict(
            status=_status_for(problems),
            checked_by=CHECKED_BY_BOTH,
            problems=problems,
        )

    async def _judge(
        self, question: str, answer: str, citations: list[ChatCitation]
    ) -> list[GateProblem] | None:
        """The reviewer's problems, or None when it could not be asked."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Answer to review:\n{answer}\n\n"
                    f"Evidence available:\n{format_evidence(citations)}"
                ),
            },
        ]

        for attempt in range(JUDGE_ATTEMPTS):
            try:
                from services.llm_chat import complete_json_object, effort_from_env

                effort = effort_from_env("JUDGE_REASONING_EFFORT", default="none")
                payload = await asyncio.to_thread(
                    complete_json_object,
                    self.llm_client,
                    messages=messages,
                    model=self.judge_model or self.llm_model or "",
                    schema={
                        "type": "object",
                        "properties": {
                            "verdict": {
                                "type": "string",
                                "enum": ["supported", "unsupported"],
                            },
                            "reason": {"type": "string"},
                            "unsupported_claims": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["verdict"],
                    },
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=JUDGE_MAX_TOKENS,
                    effort=effort,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"⚠️  Evidence reviewer attempt {attempt + 1} failed: {exc}")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That was not valid JSON matching the schema. "
                            f"Error: {exc}. Return ONLY the JSON object."
                        ),
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001 - reviewer must fail open
                print(f"⚠️  Evidence reviewer unreachable: {exc}")
                return None

            try:
                return _problems_from_payload(payload)
            except ValueError as exc:
                print(f"⚠️  Evidence reviewer attempt {attempt + 1} failed: {exc}")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That was not valid JSON matching the schema. "
                            f"Error: {exc}. Return ONLY the JSON object."
                        ),
                    }
                )

        return None

    def _complete(self, messages: list[dict[str, str]]) -> str:
        from services.llm_chat import effort_from_env, message_text, with_chat_extras

        # Judge reviews short claims; long thinking is usually wasted.
        effort = effort_from_env("JUDGE_REASONING_EFFORT", default="none")
        response = self.llm_client.chat.completions.create(
            **with_chat_extras(
                {
                    "model": self.judge_model,
                    "messages": messages,
                    "temperature": JUDGE_TEMPERATURE,
                    "max_tokens": JUDGE_MAX_TOKENS,
                    "stream": False,
                },
                effort=effort,
            )
        )
        return message_text(response.choices[0].message)


def _status_for(problems: list[GateProblem]) -> GateStatus:
    if any(problem.hard for problem in problems):
        return GateStatus.unsupported
    return GateStatus.supported


def _problems_from_payload(payload: dict) -> list[GateProblem]:
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"supported", "unsupported"}:
        raise ValueError(f"Unknown verdict {verdict!r}")
    if verdict == "supported":
        return []

    reason = str(payload.get("reason", "")).strip()
    raw_claims = payload.get("unsupported_claims") or []
    if not isinstance(raw_claims, list):
        raise ValueError("unsupported_claims must be a list")

    claims = [str(claim).strip() for claim in raw_claims if str(claim).strip()]
    detail = reason or "The reviewer found claims the evidence does not support."
    if claims:
        quoted = "; ".join(f"“{claim}”" for claim in claims[:MAX_JUDGE_CLAIMS])
        detail = f"{detail} Unsupported: {quoted}."
    return [
        GateProblem(kind=GateProblemKind.unsupported_claim, detail=detail),
    ]
