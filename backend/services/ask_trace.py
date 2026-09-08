"""Ask/Chat traces: JSONL on disk, optional Comet Opik."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
TRACE_DIR = WORKSPACE_ROOT / "data" / "traces"


def opik_configured() -> bool:
    return bool((os.getenv("OPIK_API_KEY") or "").strip())


def traces_dir() -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    return TRACE_DIR


def estimate_cost(
    prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    if prompt_tokens is None and completion_tokens is None:
        return None
    try:
        inp = float(os.getenv("USD_PER_1K_INPUT") or 0)
        out = float(os.getenv("USD_PER_1K_OUTPUT") or 0)
    except ValueError:
        return None
    prompt = prompt_tokens or 0
    completion = completion_tokens or 0
    return round((prompt / 1000.0) * inp + (completion / 1000.0) * out, 6)


def _append_jsonl(name: str, payload: dict[str, Any]) -> None:
    path = traces_dir() / f"{name}.jsonl"
    line = json.dumps(payload, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _opik_trace(name: str, *, inputs: dict, output: dict, metadata: dict) -> None:
    if not opik_configured():
        return
    try:
        from opik import Opik
    except ImportError:
        return
    try:
        client = Opik(
            project_name=os.getenv("OPIK_PROJECT_NAME", "knowledgehub-ask"),
            workspace=os.getenv("OPIK_WORKSPACE") or None,
        )
        trace = client.trace(
            name=name,
            input=inputs,
            output=output,
            metadata=metadata,
        )
        end = getattr(trace, "end", None)
        if callable(end):
            end()
    except Exception as exc:
        print(f"⚠️  Opik trace skipped: {exc}", flush=True)


def log_ask_trace(
    *,
    question: str,
    query_kind: str,
    hits: list[dict[str, Any]],
    decision: str,
    external_search: bool,
    external_status: str,
    web_urls: list[str],
    citations: list[dict[str, Any]],
    answer: str,
    latency_ms: float,
    request_id: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> None:
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "question": question,
        "query_kind": query_kind,
        "decision": decision,
        "external_search": external_search,
        "external_search_status": external_status,
        "web_urls": web_urls,
        "hits": hits[:16],
        "citations": citations[:16],
        "answer": answer[:2000],
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": estimate_cost(prompt_tokens, completion_tokens),
    }
    try:
        _append_jsonl("ask", payload)
    except OSError as exc:
        print(f"⚠️  Ask JSONL trace skipped: {exc}", flush=True)
    _opik_trace(
        "ask",
        inputs={"question": question, "external_search": external_search},
        output={"answer": answer[:2000], "citation_count": len(citations)},
        metadata={
            "request_id": request_id,
            "query_kind": query_kind,
            "decision": decision,
            "external_search_status": external_status,
            "web_urls": web_urls,
            "hits": hits[:16],
            "citations": citations[:16],
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    )


def log_chat_trace(
    *,
    question: str,
    tools_called: list[str],
    citations: list[dict[str, Any]],
    answer: str,
    latency_ms: float,
    gate_status: str | None = None,
    request_id: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> None:
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "question": question,
        "tools": tools_called,
        "gate_status": gate_status,
        "citations": citations[:16],
        "answer": answer[:2000],
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": estimate_cost(prompt_tokens, completion_tokens),
    }
    try:
        _append_jsonl("chat", payload)
    except OSError as exc:
        print(f"⚠️  Chat JSONL trace skipped: {exc}", flush=True)
    _opik_trace(
        "chat",
        inputs={"question": question},
        output={"answer": answer[:2000], "citation_count": len(citations)},
        metadata={
            "request_id": request_id,
            "tools_called": tools_called,
            "gate_status": gate_status,
            "citations": citations[:16],
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    )
