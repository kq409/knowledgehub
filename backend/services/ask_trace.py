"""Optional Comet Opik tracing for Ask. No-ops without OPIK_API_KEY."""

from __future__ import annotations

import os
from typing import Any


def opik_configured() -> bool:
    return bool((os.getenv("OPIK_API_KEY") or "").strip())


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
) -> None:
    if not opik_configured():
        return
    try:
        from opik import Opik
    except ImportError:
        return
    try:
        client = Opik(
            project_name=os.getenv("OPIK_PROJECT_NAME", "researchpilot-ask"),
            workspace=os.getenv("OPIK_WORKSPACE") or None,
        )
        trace = client.trace(
            name="ask",
            input={"question": question, "external_search": external_search},
            output={"answer": answer[:2000], "citation_count": len(citations)},
            metadata={
                "query_kind": query_kind,
                "decision": decision,
                "external_search_status": external_status,
                "web_urls": web_urls,
                "hits": hits[:16],
                "citations": citations[:16],
                "latency_ms": round(latency_ms, 1),
            },
        )
        end = getattr(trace, "end", None)
        if callable(end):
            end()
    except Exception as exc:
        print(f"⚠️  Opik trace skipped: {exc}", flush=True)
