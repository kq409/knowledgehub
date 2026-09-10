"""Public demo seed and upload/eval guards."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from services.settings import (
    demo_mode,
    demo_reset_token,
    eval_run_allowed,
    uploads_enabled,
)

UPLOADS_DISABLED = "Uploads are disabled on the public demo."
EVAL_LOCKED = "Eval runs are locked on this demo. They run in CI, not here."
RESET_DENIED = "Demo reset is not configured or the token does not match."


def require_uploads() -> None:
    if not uploads_enabled():
        raise HTTPException(status_code=403, detail=UPLOADS_DISABLED)


def require_eval_run(request: Request) -> None:
    token = request.headers.get("X-Eval-Token")
    if not eval_run_allowed(token):
        raise HTTPException(status_code=403, detail=EVAL_LOCKED)


def require_demo_reset(request: Request) -> None:
    expected = demo_reset_token()
    got = (request.headers.get("X-Demo-Reset-Token") or "").strip()
    if not expected or not got or not secrets.compare_digest(got, expected):
        raise HTTPException(status_code=403, detail=RESET_DENIED)


async def seed_demo_library(session: AsyncSession, embeddings) -> None:
    """Idempotent seed of the ACL island plus the ZXQ Ask library."""
    from eval.fixtures import seed_library
    from eval.library_island import seed_library_island

    await seed_library_island(session)
    embed_fn = getattr(embeddings, "embed_document", None)
    try:
        await seed_library(session, embed_document=embed_fn)
    except Exception as exc:
        print(f"⚠️  Demo seed with live embeddings failed ({exc}); using axes.")
        await session.rollback()
        await seed_library(session, embed_document=None)
    if demo_mode():
        print("🌱 Demo library seeded (alice/bob spaces + ZXQ papers)")
