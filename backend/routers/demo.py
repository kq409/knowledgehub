from fastapi import APIRouter, HTTPException, Request

import db
from services.demo import require_demo_reset, seed_demo_library

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.post("/reset")
async def reset_demo_library(request: Request):
    """Replace the live library with the synthetic demo seed.

    Nightly cron calls this so a public demo is not permanently vandalized.
    """
    require_demo_reset(request)
    if db.SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database is not initialized")
    embeddings = getattr(request.app.state, "embeddings", None)
    async with db.SessionLocal() as session:
        await seed_demo_library(session, embeddings)
    return {"status": "reset"}
