"""Run Ask/Chat eval suites against the production services."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

import db
from eval.fixtures import KeywordEmbeddingService, SeedResult, seed_library
from eval.graders import (
    FaithfulnessInput,
    Grade,
    abstain,
    faithfulness,
    must_include_facts,
    outcome,
    retrieval_ids,
    retrieval_titles,
    score_task,
    tool_used,
    transcript,
)
from eval.graders import (
    citations as grade_citations,
)
from eval.library_island import seed_library_island
from eval.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    suite_pass_at_k,
    suite_pass_hat_k,
)
from eval.store import new_run_id, write_run
from schemas import (
    AskRequest,
    ChatCitation,
    ChatMessage,
    ChatRequest,
    ChatRole,
)
from services.agent.gate import EvidenceGate
from services.agent.loop import ResearchAgent
from services.agent.permissions import subagent_policy
from services.ask import AskService
from services.embeddings import EmbeddingService
from services.identity import eval_zxq_identity, load_identity
from services.library_records import parse_content_type, search_records
from services.retrieval import search

SUITES_DIR = Path(__file__).resolve().parent / "suites"

SUITE_ALIASES: dict[str, tuple[str, ...]] = {
    "ask": ("ask-regression", "ask-quality"),
    "chat": ("chat-regression", "chat-quality"),
    "all": (
        "ask-regression",
        "ask-quality",
        "chat-regression",
        "chat-quality",
        "acl-regression",
        "search-regression",
        "search-quality",
        "discipline-regression",
    ),
}
KNOWN_SUITES = frozenset(
    {
        "ask",
        "chat",
        "all",
        "ask-regression",
        "ask-quality",
        "chat-regression",
        "chat-quality",
        "acl-regression",
        "search-regression",
        "search-quality",
        "discipline-regression",
    }
)
CHAT_SUITES = frozenset({"chat", "chat-regression", "chat-quality"})
ACL_SUITES = frozenset({"acl-regression"})
SEARCH_SUITES = frozenset({"search-regression", "search-quality"})
DISCIPLINE_SUITES = frozenset({"discipline-regression"})


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubMessage(content)


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]
        self.usage = None


class StubLLM:
    """Deterministic Ask completions so retrieval suites run without a model."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> _StubResponse:
        return _StubResponse("ZXQELLA7 transfers sparse models across tasks [1].")


def expand_suite_names(name: str) -> list[str]:
    return list(SUITE_ALIASES.get(name, (name,)))


def load_suite(name: str) -> dict[str, Any]:
    path = SUITES_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown eval suite {name!r} ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def _unique_ids(hits: list) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        value = str(hit.source_id)
        if value in seen:
            continue
        seen.add(value)
        ids.append(value)
    return ids


def _unique_titles(hits: list) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        title = hit.title
        if title in seen:
            continue
        seen.add(title)
        titles.append(title)
    return titles


def _registered(citations: list) -> set[int]:
    indices: set[int] = set()
    for item in citations:
        if isinstance(item, dict):
            indices.add(int(item["index"]))
        else:
            indices.add(int(item.index))
    return indices


async def _grade_specs(
    specs: list[dict[str, Any]],
    *,
    retrieved_titles: list[str],
    insufficient_evidence: bool,
    suggest_external_search: bool,
    answer: str,
    citation_objs: list,
    tools_called: list[str],
    gate_status: str | None,
    question: str,
    use_llm: bool,
    gate: EvidenceGate | None,
    retrieved_ids: list[str] | None = None,
) -> list[Grade]:
    grades: list[Grade] = []
    registered = _registered(citation_objs)
    tool_count = len(tools_called)
    for spec in specs:
        kind = spec.get("type")
        if kind == "retrieval_titles":
            grades.append(
                retrieval_titles(
                    retrieved_titles,
                    must_include=spec.get("must_include") or [],
                    must_exclude=spec.get("must_exclude") or [],
                    k=int(spec.get("k") or 8),
                )
            )
        elif kind == "retrieval_ids":
            grades.append(
                retrieval_ids(
                    retrieved_ids or [],
                    must_include=spec.get("must_include") or [],
                    must_exclude=spec.get("must_exclude") or [],
                )
            )
        elif kind == "abstain":
            grades.append(
                abstain(
                    insufficient_evidence=insufficient_evidence,
                    suggest_external_search=suggest_external_search,
                    expect=str(spec.get("expect") or "insufficient_evidence"),
                )
            )
        elif kind == "citations":
            grades.append(
                grade_citations(
                    answer,
                    registered,
                    require_indices=bool(spec.get("require_indices")),
                    tool_calls_made=tool_count,
                )
            )
        elif kind == "tool_used":
            grades.append(
                tool_used(
                    tools_called,
                    spec.get("any_of"),
                    none_of=spec.get("none_of"),
                    required=bool(spec.get("required", False)),
                )
            )
        elif kind == "must_include_facts":
            grades.append(
                must_include_facts(
                    answer,
                    list(spec.get("facts") or []),
                    required=bool(spec.get("required", True)),
                )
            )
        elif kind == "transcript":
            grades.append(
                transcript(
                    tools_called=tools_called,
                    answer=answer,
                    registered_indices=registered,
                    max_tool_calls=spec.get("max_tool_calls"),
                )
            )
        elif kind == "outcome":
            grades.append(
                outcome(
                    answer,
                    contains=spec.get("contains"),
                    contains_any=spec.get("contains_any"),
                    gate_status=gate_status,
                    expect_gate=spec.get("expect_gate"),
                    excludes=spec.get("excludes"),
                    required=bool(spec.get("required", True)),
                )
            )
        elif kind == "faithfulness":
            if not use_llm or gate is None:
                continue
            parsed: list[ChatCitation] = []
            for item in citation_objs:
                if isinstance(item, ChatCitation):
                    parsed.append(item)
                elif isinstance(item, dict):
                    parsed.append(ChatCitation.model_validate(item))
            grades.append(
                await faithfulness(
                    FaithfulnessInput(
                        question=question,
                        answer=answer,
                        citations=parsed,
                        tool_calls_made=max(tool_count, 1),
                    ),
                    gate,
                )
            )
    return grades


async def _run_ask_task(
    session: AsyncSession,
    ask: AskService,
    task: dict[str, Any],
    seed: SeedResult,
    *,
    use_llm: bool,
    gate: EvidenceGate | None,
) -> dict[str, Any]:
    query = task["query"]
    query_embedding = await asyncio.to_thread(ask.embeddings.embed_query, query)
    spaces = eval_zxq_identity().space_ids
    hits = await search(
        session,
        query_embedding,
        query_text=query,
        paper_ids=seed.paper_ids,
        space_ids=spaces,
    )
    titles = _unique_titles(hits)
    payload = AskRequest(question=query, paper_ids=seed.paper_ids)
    result = await ask.ask(session, payload, space_ids=spaces)
    grades = await _grade_specs(
        task.get("graders") or [],
        retrieved_titles=titles,
        insufficient_evidence=result.insufficient_evidence,
        suggest_external_search=result.suggest_external_search,
        answer=result.answer,
        citation_objs=result.citations,
        tools_called=[],
        gate_status=None,
        question=query,
        use_llm=use_llm,
        gate=gate,
    )
    score, passed = score_task(grades)
    k = 8
    relevant = []
    for spec in task.get("graders") or []:
        if spec.get("type") == "retrieval_titles":
            relevant = list(spec.get("must_include") or [])
            k = int(spec.get("k") or 8)
            break
    metrics = {}
    if relevant:
        metrics = {
            "precision_at_k": precision_at_k(titles, relevant, k),
            "recall_at_k": recall_at_k(titles, relevant, k),
            "mrr": mean_reciprocal_rank(titles, relevant),
        }
    return {
        "task_id": task["id"],
        "kind": task.get("kind"),
        "query": query,
        "answer": result.answer,
        "insufficient_evidence": result.insufficient_evidence,
        "suggest_external_search": result.suggest_external_search,
        "retrieved_titles": titles,
        "citations": [item.model_dump(mode="json") for item in result.citations],
        "grades": [grade.as_dict() for grade in grades],
        "score": score,
        "passed": passed,
        "metrics": metrics,
        "request_id": result.request_id,
        "source": "ask",
    }


async def _run_chat_task(
    session: AsyncSession,
    agent: ResearchAgent,
    task: dict[str, Any],
    *,
    use_llm: bool,
    gate: EvidenceGate | None,
) -> dict[str, Any]:
    query = task["query"]
    request = ChatRequest(messages=[ChatMessage(role=ChatRole.user, content=query)])
    tools_called: list[str] = []
    answer = ""
    citation_objs: list = []
    gate_status: str | None = None
    events: list[dict[str, Any]] = []
    async for event in agent.run(session, request):
        payload = {"type": event.type.value, **event.data}
        events.append(payload)
        if event.type.value == "tool_call":
            name = event.data.get("name")
            if isinstance(name, str):
                tools_called.append(name)
        elif event.type.value == "token":
            answer += str(event.data.get("text") or "")
        elif event.type.value == "citations":
            citation_objs = event.data.get("citations") or []
        elif event.type.value == "verdict":
            status = event.data.get("status")
            if status and status != "retrying":
                gate_status = str(status)
    grades = await _grade_specs(
        task.get("graders") or [],
        retrieved_titles=[],
        insufficient_evidence=False,
        suggest_external_search=False,
        answer=answer,
        citation_objs=citation_objs,
        tools_called=tools_called,
        gate_status=gate_status,
        question=query,
        use_llm=use_llm,
        gate=gate,
    )
    score, passed = score_task(grades)
    return {
        "task_id": task["id"],
        "kind": task.get("kind"),
        "query": query,
        "answer": answer,
        "tools_called": tools_called,
        "gate_status": gate_status,
        "citations": citation_objs,
        "events": events,
        "grades": [grade.as_dict() for grade in grades],
        "score": score,
        "passed": passed,
        "source": "agent",
    }


async def _run_acl_task(
    session: AsyncSession,
    ask: AskService,
    task: dict[str, Any],
    *,
    use_llm: bool,
    gate: EvidenceGate | None,
) -> dict[str, Any]:
    query = task["query"]
    identity = await load_identity(session, str(task.get("user") or "alice"))
    record_ids = None
    if task.get("record_ids") is not None:
        record_ids = [uuid.UUID(str(item)) for item in task["record_ids"]]
    query_embedding = await asyncio.to_thread(ask.embeddings.embed_query, query)
    hits = await search(
        session,
        query_embedding,
        query_text=query,
        space_ids=identity.space_ids,
        record_ids=record_ids,
    )
    titles = _unique_titles(hits)
    ids = _unique_ids(hits)
    payload = AskRequest(question=query, paper_ids=record_ids)
    result = await ask.ask(session, payload, space_ids=identity.space_ids)
    grades = await _grade_specs(
        task.get("graders") or [],
        retrieved_titles=titles,
        retrieved_ids=ids,
        insufficient_evidence=result.insufficient_evidence,
        suggest_external_search=result.suggest_external_search,
        answer=result.answer,
        citation_objs=result.citations,
        tools_called=[],
        gate_status=None,
        question=query,
        use_llm=use_llm,
        gate=gate,
    )
    score, passed = score_task(grades)
    return {
        "task_id": task["id"],
        "kind": task.get("kind"),
        "query": query,
        "user": identity.user_id,
        "answer": result.answer,
        "insufficient_evidence": result.insufficient_evidence,
        "retrieved_titles": titles,
        "retrieved_ids": ids,
        "citations": [item.model_dump(mode="json") for item in result.citations],
        "grades": [grade.as_dict() for grade in grades],
        "score": score,
        "passed": passed,
        "source": "acl",
    }


async def _run_search_task(
    session: AsyncSession,
    embeddings: EmbeddingService | KeywordEmbeddingService,
    task: dict[str, Any],
    *,
    use_llm: bool,
    use_real_embeddings: bool,
    gate: EvidenceGate | None,
) -> dict[str, Any]:
    query = task["query"]
    identity = await load_identity(session, str(task.get("user") or "alice"))
    space_ids = identity.space_ids
    space_raw = task.get("space_id")
    if space_raw:
        wanted = uuid.UUID(str(space_raw))
        space_ids = frozenset({wanted}) & identity.space_ids
    content_type = None
    if task.get("content_type"):
        content_type = parse_content_type(str(task["content_type"]))
    record_ids = None
    if task.get("record_ids") is not None:
        record_ids = [uuid.UUID(str(item)) for item in task["record_ids"]]
    backend = str(task.get("search") or "catalog")
    if backend == "rag":
        query_embedding = await asyncio.to_thread(embeddings.embed_query, query)
        hits = await search(
            session,
            query_embedding,
            query_text=query,
            space_ids=space_ids,
            record_ids=record_ids,
        )
        titles = _unique_titles(hits)
        ids = _unique_ids(hits)
    else:
        page = await search_records(
            session,
            query,
            space_ids=space_ids,
            content_type=content_type,
            record_ids=record_ids,
            limit=int(task.get("limit") or 50),
            offset=int(task.get("offset") or 0),
        )
        titles = [item.title for item in page.items]
        ids = [str(item.id) for item in page.items]
    grades = await _grade_specs(
        task.get("graders") or [],
        retrieved_titles=titles,
        retrieved_ids=ids,
        insufficient_evidence=len(ids) == 0,
        suggest_external_search=False,
        answer="",
        citation_objs=[],
        tools_called=[],
        gate_status=None,
        question=query,
        use_llm=use_llm,
        gate=gate,
    )
    score, passed = score_task(grades)
    return {
        "task_id": task["id"],
        "kind": task.get("kind"),
        "query": query,
        "user": identity.user_id,
        "retrieved_titles": titles,
        "retrieved_ids": ids,
        "grades": [grade.as_dict() for grade in grades],
        "score": score,
        "passed": passed,
        "source": "search",
        "retrieval_backend": backend,
        "embedding_mode": (
            "real"
            if use_real_embeddings and backend == "rag"
            else ("none" if backend == "catalog" else "keyword")
        ),
    }


async def grade_reference(
    task: dict[str, Any],
    *,
    use_llm: bool = False,
    gate: EvidenceGate | None = None,
) -> dict[str, Any]:
    """Score the task's reference payload. Proves the task is solvable."""
    ref = dict(task.get("reference") or {})
    if not ref:
        raise ValueError(f"task {task.get('id')!r} has no reference")
    citations = list(ref.get("citations") or [])
    tools_called = list(ref.get("tools_called") or [])
    grades = await _grade_specs(
        task.get("graders") or [],
        retrieved_titles=list(ref.get("retrieved_titles") or []),
        retrieved_ids=[str(item) for item in (ref.get("retrieved_ids") or [])],
        insufficient_evidence=bool(ref.get("insufficient_evidence", False)),
        suggest_external_search=bool(ref.get("suggest_external_search", False)),
        answer=str(ref.get("answer") or ""),
        citation_objs=citations,
        tools_called=tools_called,
        gate_status=ref.get("gate_status"),
        question=str(task.get("query") or ""),
        use_llm=use_llm,
        gate=gate,
    )
    score, passed = score_task(grades)
    return {
        "task_id": task["id"],
        "kind": task.get("kind"),
        "query": task.get("query"),
        "answer": ref.get("answer") or "",
        "tools_called": tools_called,
        "gate_status": ref.get("gate_status"),
        "citations": citations,
        "events": list(ref.get("events") or []),
        "retrieved_titles": list(ref.get("retrieved_titles") or []),
        "insufficient_evidence": bool(ref.get("insufficient_evidence", False)),
        "suggest_external_search": bool(ref.get("suggest_external_search", False)),
        "grades": [grade.as_dict() for grade in grades],
        "score": score,
        "passed": passed,
        "source": "reference",
    }


def _kind_groups(trials: list[dict[str, Any]], kind: str) -> list[list[bool]]:
    by_task: dict[str, list[bool]] = {}
    for trial in trials:
        if trial.get("kind") != kind:
            continue
        by_task.setdefault(trial["task_id"], []).append(bool(trial["passed"]))
    return list(by_task.values())


def _summarize(suite_name: str, trials: list[dict[str, Any]], default_k: int) -> dict:
    by_task: dict[str, list[bool]] = {}
    for trial in trials:
        by_task.setdefault(trial["task_id"], []).append(bool(trial["passed"]))
    groups = list(by_task.values())
    k = max(1, default_k)
    retrieval_p: list[float] = []
    retrieval_r: list[float] = []
    retrieval_mrr: list[float] = []
    for trial in trials:
        metrics = trial.get("metrics") or {}
        if "precision_at_k" in metrics:
            retrieval_p.append(metrics["precision_at_k"])
            retrieval_r.append(metrics["recall_at_k"])
            retrieval_mrr.append(metrics["mrr"])
    regression = [trial for trial in trials if trial.get("kind") == "regression"]
    regression_groups = _kind_groups(trials, "regression")
    quality_groups = _kind_groups(trials, "capability")
    summary = {
        "suite": suite_name,
        "tasks": len(by_task),
        "trials": len(trials),
        "pass_at_1": suite_pass_at_k(groups, 1) if groups else 0.0,
        "pass_hat_k": (
            suite_pass_hat_k(groups, k)
            if k > 1 and groups
            else suite_pass_at_k(groups, 1) if groups else 0.0
        ),
        "pass_rate": (
            sum(1 for trial in trials if trial["passed"]) / len(trials)
            if trials
            else 0.0
        ),
        "regression_tasks": len(regression_groups),
        "quality_tasks": len(quality_groups),
        "regression_pass_rate": (
            sum(1 for trial in regression if trial["passed"]) / len(regression)
            if regression
            else 1.0
        ),
        "regression_failed": any(not trial["passed"] for trial in regression),
        "quality_pass_at_1": (
            suite_pass_at_k(quality_groups, 1) if quality_groups else None
        ),
        "quality_pass_hat_k": (
            suite_pass_hat_k(quality_groups, k)
            if quality_groups and k > 1
            else suite_pass_at_k(quality_groups, 1) if quality_groups else None
        ),
        "gate": (
            "fail"
            if any(not trial["passed"] for trial in regression)
            or (
                "regression" in suite_name
                and any(not trial["passed"] for trial in trials)
            )
            else "pass"
        ),
    }
    backends = sorted(
        {
            str(trial.get("retrieval_backend"))
            for trial in trials
            if trial.get("retrieval_backend")
        }
    )
    modes = sorted(
        {
            str(trial.get("embedding_mode"))
            for trial in trials
            if trial.get("embedding_mode")
        }
    )
    if backends:
        summary["retrieval_backend"] = ",".join(backends)
    if modes:
        summary["embedding_mode"] = ",".join(modes)
        if "keyword" in modes and "search" in suite_name:
            summary["embedding_note"] = (
                "keyword-axis embeddings prove fixture/grader agreement, not retrieval SOTA"
            )
    if retrieval_p:
        summary["precision_at_k"] = sum(retrieval_p) / len(retrieval_p)
        summary["recall_at_k"] = sum(retrieval_r) / len(retrieval_r)
        summary["mrr"] = sum(retrieval_mrr) / len(retrieval_mrr)
    return summary


def _print_one_summary(summary: dict[str, Any]) -> None:
    print(f"  suite={summary.get('suite')}")
    for key, value in summary.items():
        if key == "suite":
            continue
        if isinstance(value, float):
            print(f"    {key}: {value:.3f}")
        else:
            print(f"    {key}: {value}")


def _print_summary(manifest: dict[str, Any], trials: list[dict[str, Any]]) -> None:
    print(f"Eval run {manifest['id']}  suite={manifest['suite']}")
    for summary in manifest.get("suites") or []:
        _print_one_summary(summary)
    print()
    for trial in trials:
        mark = "PASS" if trial["passed"] else "FAIL"
        kind = trial.get("kind") or ""
        source = trial.get("source") or ""
        extra = " ".join(part for part in (kind, source) if part)
        print(f"  [{mark}] {trial['task_id']}  score={trial['score']:.2f}  {extra}")
        for grade in trial.get("grades") or []:
            flag = "ok" if grade["passed"] else "no"
            req = "" if grade.get("required", True) else " (diagnostic)"
            print(f"       {flag} {grade['name']}{req}: {grade['detail']}")


def regression_gate_failed(summaries: list[dict[str, Any]]) -> bool:
    """True when a regression task failed — the CD / CI gate."""
    return any(
        bool(item.get("regression_failed")) or item.get("gate") == "fail"
        for item in summaries
    )


async def run_suites(
    names: list[str],
    *,
    use_llm: bool,
    session: AsyncSession,
    embeddings: EmbeddingService | KeywordEmbeddingService,
    llm_client: Any,
    llm_model: str,
    use_real_embeddings: bool = False,
) -> dict[str, Any]:
    from eval.discipline_island import cleanup_discipline_island, seed_discipline_island
    from eval.fixtures import cleanup_eval_fixtures
    from eval.library_island import cleanup_library_island

    embed_fn = (
        embeddings.embed_document if hasattr(embeddings, "embed_document") else None
    )
    if use_real_embeddings and hasattr(embeddings, "embed_document"):
        embed_fn = embeddings.embed_document
    elif not use_llm:
        embed_fn = None
    seed = await seed_library(session, embed_document=embed_fn)
    ask = AskService(
        llm_client=llm_client,
        llm_model=llm_model,
        embeddings=embeddings,
    )
    gate = EvidenceGate(
        llm_client=llm_client if use_llm else None,
        llm_model=llm_model,
        base_url=os.getenv("LLM_BASE_URL"),
        mode=None,
    )
    run_id = new_run_id()
    all_trials: list[dict[str, Any]] = []
    suite_summaries: list[dict[str, Any]] = []
    try:
        for name in names:
            suite = load_suite(name)
            is_acl = name in ACL_SUITES
            is_search = name in SEARCH_SUITES
            is_discipline = name in DISCIPLINE_SUITES
            is_chat = name in CHAT_SUITES or name.startswith("chat")
            trials_n = int(suite.get("trials") or (3 if name == "chat-quality" else 1))
            tasks = list(suite.get("tasks") or [])
            suite_trials: list[dict[str, Any]] = []
            run_agent = is_chat and use_llm
            grade_refs = is_chat and not use_llm
            if is_acl or is_search or is_discipline:
                await seed_library_island(session)
            if is_discipline:
                await seed_discipline_island(session)
            if is_discipline:
                for task in tasks:
                    if task.get("search"):
                        trial = await _run_search_task(
                            session,
                            embeddings,
                            task,
                            use_llm=use_llm,
                            use_real_embeddings=use_real_embeddings,
                            gate=gate,
                        )
                    else:
                        trial = await _run_acl_task(
                            session, ask, task, use_llm=use_llm, gate=gate
                        )
                    trial["trial_index"] = 0
                    trial["suite"] = name
                    suite_trials.append(trial)
            elif is_search:
                for task in tasks:
                    trial = await _run_search_task(
                        session,
                        embeddings,
                        task,
                        use_llm=use_llm,
                        use_real_embeddings=use_real_embeddings,
                        gate=gate,
                    )
                    trial["trial_index"] = 0
                    trial["suite"] = name
                    suite_trials.append(trial)
            elif is_acl:
                for task in tasks:
                    trial = await _run_acl_task(
                        session, ask, task, use_llm=use_llm, gate=gate
                    )
                    trial["trial_index"] = 0
                    trial["suite"] = name
                    suite_trials.append(trial)
            elif run_agent:
                agent = ResearchAgent(
                    llm_client=llm_client,
                    llm_model=llm_model,
                    embeddings=embeddings,
                    gate=gate,
                    policy=subagent_policy(),
                )
                for task in tasks:
                    for index in range(trials_n):
                        trial = await _run_chat_task(
                            session, agent, task, use_llm=use_llm, gate=gate
                        )
                        trial["trial_index"] = index
                        trial["suite"] = name
                        suite_trials.append(trial)
            elif grade_refs:
                print(
                    f"Grading {name} references (pass --llm to run the production agent)."
                )
                for task in tasks:
                    trial = await grade_reference(task, use_llm=False, gate=None)
                    trial["trial_index"] = 0
                    trial["suite"] = name
                    suite_trials.append(trial)
            else:
                for task in tasks:
                    for index in range(trials_n):
                        trial = await _run_ask_task(
                            session,
                            ask,
                            task,
                            seed,
                            use_llm=use_llm,
                            gate=gate,
                        )
                        trial["trial_index"] = index
                        trial["suite"] = name
                        suite_trials.append(trial)
            summary = _summarize(name, suite_trials, trials_n)
            suite_summaries.append(summary)
            all_trials.extend(suite_trials)
    finally:
        await cleanup_eval_fixtures(session)
        await cleanup_library_island(session)
        await cleanup_discipline_island(session)

    manifest = {
        "id": run_id,
        "suite": ",".join(names),
        "use_llm": use_llm,
        "model": llm_model,
        "created_at": run_id,
        "summary": suite_summaries[0] if len(suite_summaries) == 1 else suite_summaries,
        "suites": suite_summaries,
    }
    write_run(run_id, manifest, all_trials)
    _print_summary(manifest, all_trials)
    print(f"\nWrote {Path('eval/runs') / run_id}")
    return {"manifest": manifest, "trials": all_trials}


def _build_clients(use_llm: bool, use_real_embeddings: bool):
    load_dotenv()
    if use_llm or use_real_embeddings:
        embeddings = EmbeddingService(
            OpenAI(
                base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL"),
                api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY"),
            ),
            os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        )
    else:
        embeddings = KeywordEmbeddingService()
    if use_llm:
        llm = OpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
        )
        model = os.getenv("LLM_MODEL") or "unknown"
    else:
        llm = StubLLM()
        model = "stub"
    return embeddings, llm, model


async def _async_check_references(names: list[str]) -> int:
    failed = 0
    for name in names:
        suite = load_suite(name)
        print(f"References {name}")
        for task in suite.get("tasks") or []:
            trial = await grade_reference(task)
            mark = "PASS" if trial["passed"] else "FAIL"
            print(f"  [{mark}] {trial['task_id']}  score={trial['score']:.2f}")
            if not trial["passed"]:
                failed += 1
                for grade in trial.get("grades") or []:
                    if not grade["passed"] and grade.get("required", True):
                        print(f"       no {grade['name']}: {grade['detail']}")
    return 1 if failed else 0


async def _async_main(
    names: list[str], use_llm: bool, use_real_embeddings: bool
) -> int:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 1
    db.init_db(database_url)
    if db.SessionLocal is None:
        print("Could not open a database session", file=sys.stderr)
        return 1
    real_vectors = use_llm or use_real_embeddings
    embeddings, llm, model = _build_clients(use_llm, use_real_embeddings=real_vectors)
    result: dict[str, Any] | None = None
    try:
        async with db.SessionLocal() as session:
            result = await run_suites(
                names,
                use_llm=use_llm,
                session=session,
                embeddings=embeddings,
                llm_client=llm,
                llm_model=model,
                use_real_embeddings=real_vectors,
            )
    finally:
        await db.close_db()
    if result is None:
        return 1
    summaries = list(result["manifest"].get("suites") or [])
    return 1 if regression_gate_failed(summaries) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run KnowledgeHub eval suites")
    parser.add_argument(
        "--suite",
        default="ask-regression",
        choices=tuple(sorted(KNOWN_SUITES)),
        help="Which suite to run. 'ask'/'chat'/'all' expand to split suites.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Call the configured LLM (needed for Chat agent; optional for Ask)",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help=(
            "Use the configured embedding host (Ollama/nomic). Required to treat "
            "search-quality as a real-vector run. Keyword-axis is not retrieval SOTA."
        ),
    )
    parser.add_argument(
        "--check-references",
        action="store_true",
        help="Only grade each task's reference payload (no model, no retrieval)",
    )
    args = parser.parse_args(argv)
    names = expand_suite_names(args.suite)
    if args.check_references:
        return asyncio.run(_async_check_references(names))
    return asyncio.run(_async_main(names, args.llm, args.embeddings))


if __name__ == "__main__":
    raise SystemExit(main())
