"""Read-only MCP server wrapping ResearchPilot library tools.

Run from backend/: ``uv run python -m mcp_server``
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

import db
from services.agent.tools import (
    SUBAGENT_TOOL_NAMES,
    TOOL_HANDLERS,
    CitationRegistry,
    ToolContext,
    ToolError,
)
from services.embeddings import EmbeddingService

load_dotenv()


def _embeddings() -> EmbeddingService:
    base = os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL")
    key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or "ollama"
    model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    return EmbeddingService(OpenAI(base_url=base, api_key=key), model)


async def invoke_readonly_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    session: Any,
    embeddings: Any,
) -> str:
    if name not in SUBAGENT_TOOL_NAMES:
        raise ToolError(f"{name} is not exposed over MCP")
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ToolError(f"{name} has no implementation")
    ctx = ToolContext(
        session=session,
        embeddings=embeddings,
        registry=CitationRegistry(),
    )
    result = await handler(ctx, **(arguments or {}))
    return result.content


def _ensure_db() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    if db.SessionLocal is None:
        db.init_db(database_url)


def create_mcp():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("researchpilot")
    embeddings = _embeddings()

    async def _call(name: str, **kwargs: Any) -> str:
        _ensure_db()
        assert db.SessionLocal is not None
        async with db.SessionLocal() as session:
            return await invoke_readonly_tool(
                name, kwargs, session=session, embeddings=embeddings
            )

    @mcp.tool()
    async def search_library(query: str, top_k: int | None = None) -> str:
        payload: dict[str, Any] = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        return await _call("search_library", **payload)

    @mcp.tool()
    async def list_papers() -> str:
        return await _call("list_papers")

    @mcp.tool()
    async def list_notes(source_type: str | None = None) -> str:
        payload: dict[str, Any] = {}
        if source_type:
            payload["source_type"] = source_type
        return await _call("list_notes", **payload)

    @mcp.tool()
    async def read_paper(
        paper_id: str, start_index: int = 0, limit: int | None = None
    ) -> str:
        payload: dict[str, Any] = {"paper_id": paper_id, "start_index": start_index}
        if limit is not None:
            payload["limit"] = limit
        return await _call("read_paper", **payload)

    @mcp.tool()
    async def read_note(note_id: str) -> str:
        return await _call("read_note", note_id=note_id)

    @mcp.tool()
    async def list_skills() -> str:
        return await _call("list_skills")

    @mcp.tool()
    async def load_skill(name: str) -> str:
        return await _call("load_skill", name=name)

    @mcp.tool()
    async def memory_search(query: str) -> str:
        return await _call("memory_search", query=query)

    return mcp


def main() -> None:
    mcp = create_mcp()
    mcp.run()


if __name__ == "__main__":
    main()
