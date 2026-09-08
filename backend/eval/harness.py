"""Run Ask/Chat eval suites against the production services."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
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
    outcome,
    retrieval_titles,
    score_task,
    tool_used,
    transcript,
)
from eval.graders import (
    citations as grade_citations,
)
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
from services.retrieval import search

SUITES_DIR = Path(__file__).resolve().parent / "suites"


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


def load_suite(name: str) -> dict[str, Any]:
    path = SUITES_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


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
            grades.append(tool_used(tools_called, list(spec.get("any_of") or [])))
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
    hits = await search(
        session,
        query_embedding,
        query_text=query,
        paper_ids=seed.paper_ids,
    )
    titles = _unique_titles(hits)
    payload = AskRequest(question=query, paper_ids=seed.paper_ids)
    result = await ask.ask(session, payload)
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
    }


def _summarize(suite_name: str, trials: list[dict[str, Any]], default_k: int) -> dict:
    by_task: dict[str, list[bool]] = {}
    for trial in trials:
        by_task.setdefault(trial["task_id"], []).append(bool(trial["passed"]))
    groups = list(by_task.values())
    k = default_k
    retrieval_p: list[float] = []
    retrieval_r: list[float] = []
    retrieval_mrr: list[float] = []
    for trial in trials:
        metrics = trial.get("metrics") or {}
        if "precision_at_k" in metrics:
            retrieval_p.append(metrics["precision_at_k"])
            retrieval_r.append(metrics["recall_at_k"])
            retrieval_mrr.append(metrics["mrr"])
    summary = {
        "suite": suite_name,
        "tasks": len(by_task),
        "trials": len(trials),
        "pass_at_1": suite_pass_at_k(groups, 1),
        "pass_hat_k": (
            suite_pass_hat_k(groups, k) if k > 1 else suite_pass_at_k(groups, 1)
        ),
        "pass_rate": (
            sum(1 for trial in trials if trial["passed"]) / len(trials)
            if trials
            else 0.0
        ),
    }
    if retrieval_p:
        summary["precision_at_k"] = sum(retrieval_p) / len(retrieval_p)
        summary["recall_at_k"] = sum(retrieval_r) / len(retrieval_r)
        summary["mrr"] = sum(retrieval_mrr) / len(retrieval_mrr)
    return summary


def _print_summary(manifest: dict[str, Any], trials: list[dict[str, Any]]) -> None:
    print(f"Eval run {manifest['id']}  suite={manifest['suite']}")
    summary = manifest["summary"]
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()
    for trial in trials:
        mark = "PASS" if trial["passed"] else "FAIL"
        print(f"  [{mark}] {trial['task_id']}  score={trial['score']:.2f}")
        for grade in trial.get("grades") or []:
            flag = "ok" if grade["passed"] else "no"
            print(f"       {flag} {grade['name']}: {grade['detail']}")


async def run_suites(
    names: list[str],
    *,
    use_llm: bool,
    session: AsyncSession,
    embeddings: EmbeddingService | KeywordEmbeddingService,
    llm_client: Any,
    llm_model: str,
) -> dict[str, Any]:
    from eval.fixtures import cleanup_eval_fixtures

    embed_fn = (
        embeddings.embed_document if hasattr(embeddings, "embed_document") else None
    )
    if not use_llm:
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
            trials_n = int(suite.get("trials") or (3 if name == "chat" else 1))
            tasks = list(suite.get("tasks") or [])
            suite_trials: list[dict[str, Any]] = []
            if name == "chat":
                if not use_llm:
                    print("Skipping chat suite (pass --llm to run against the agent).")
                    continue
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


async def _async_main(names: list[str], use_llm: bool) -> int:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 1
    db.init_db(database_url)
    if db.SessionLocal is None:
        print("Could not open a database session", file=sys.stderr)
        return 1
    embeddings, llm, model = _build_clients(use_llm, use_real_embeddings=use_llm)
    try:
        async with db.SessionLocal() as session:
            await run_suites(
                names,
                use_llm=use_llm,
                session=session,
                embeddings=embeddings,
                llm_client=llm,
                llm_model=model,
            )
    finally:
        await db.close_db()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run KnowledgeHub eval suites")
    parser.add_argument(
        "--suite",
        default="ask",
        choices=("ask", "chat", "all"),
        help="Which suite to run",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Call the configured LLM (needed for Chat; optional for Ask)",
    )
    args = parser.parse_args(argv)
    names = ["ask", "chat"] if args.suite == "all" else [args.suite]
    return asyncio.run(_async_main(names, args.llm))


if __name__ == "__main__":
    raise SystemExit(main())
