from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from services.settings import (
    demo_mode,
    eval_run_allowed,
    git_sha,
    grobid_enabled,
    uploads_enabled,
    whisper_enabled,
)

router = APIRouter(tags=["health"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.get("/api/healthz")
async def healthz():
    """Process is up. Load balancers can hit this without a database."""
    return {"status": "ok"}


@router.get("/api/readyz")
async def readyz(session: SessionDep):
    """Database reachable and the agent package imports. Does not ping Ollama."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Database is not reachable"
        ) from exc
    try:
        from services.agent.loop import ResearchAgent  # noqa: F401
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Agent harness failed to import"
        ) from exc
    return {"status": "ok", "database": True, "agent": True}


@router.get("/api/status")
async def get_status(request: Request):
    import os

    ready = getattr(request.app.state, "agent", None) is not None
    payload = {
        "status": "ready" if ready else "initializing",
        "demo_mode": demo_mode(),
        "whisper_enabled": whisper_enabled(),
        "grobid_enabled": grobid_enabled(),
        "uploads_enabled": uploads_enabled(),
        "eval_run_allowed": eval_run_allowed(None),
        "llm_model": os.getenv("LLM_MODEL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "whisper_model": os.getenv("WHISPER_MODEL") if whisper_enabled() else None,
        "git_sha": git_sha(),
        "replicas": 1,
        "notice": (
            "Demo ACL via X-User-Id (alice/bob), not SSO. Single replica: "
            "approvals stay in this process."
            if demo_mode()
            else None
        ),
    }
    if not demo_mode():
        payload["llm_base_url"] = os.getenv("LLM_BASE_URL")
    return payload
