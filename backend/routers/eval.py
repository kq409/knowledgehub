from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from eval.store import list_runs, load_run

router = APIRouter(prefix="/api/eval", tags=["eval"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.get("/runs")
async def get_runs():
    return list_runs()


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    payload = load_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return payload


@router.post("/run")
async def trigger_run(
    request: Request,
    session: SessionDep,
    suite: str = Query(default="ask"),
    use_llm: bool = Query(default=False),
):
    if suite not in {"ask", "chat", "all"}:
        raise HTTPException(status_code=400, detail="suite must be ask, chat, or all")
    if suite in {"chat", "all"} and not use_llm:
        raise HTTPException(
            status_code=400,
            detail="Chat eval requires use_llm=true",
        )
    from eval.fixtures import KeywordEmbeddingService
    from eval.harness import StubLLM, run_suites

    names = ["ask", "chat"] if suite == "all" else [suite]
    ask = getattr(request.app.state, "ask", None)
    if use_llm:
        if ask is None:
            raise HTTPException(status_code=503, detail="Ask service not ready")
        embeddings = request.app.state.embeddings
        llm_client = ask.llm_client
        llm_model = ask.llm_model
    else:
        embeddings = KeywordEmbeddingService()
        llm_client = StubLLM()
        llm_model = "stub"
    result = await run_suites(
        names,
        use_llm=use_llm,
        session=session,
        embeddings=embeddings,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    return result["manifest"]
