from fastapi import APIRouter, HTTPException, Query, Request

import db
from eval.store import list_runs, load_run

router = APIRouter(prefix="/api/eval", tags=["eval"])


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
    suite: str = Query(default="ask-regression"),
    use_llm: bool = Query(default=False),
):
    from eval.fixtures import KeywordEmbeddingService
    from eval.harness import KNOWN_SUITES, StubLLM, expand_suite_names, run_suites
    from services.demo import require_eval_run

    require_eval_run(request)

    if suite not in KNOWN_SUITES:
        raise HTTPException(
            status_code=400,
            detail=(
                "suite must be a known eval suite "
                "(ask/chat/acl/search/discipline regression or quality, or all)"
            ),
        )
    names = expand_suite_names(suite)
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
    if db.SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database is not initialized")
    async with db.SessionLocal() as session:
        result = await run_suites(
            names,
            use_llm=use_llm,
            session=session,
            embeddings=embeddings,
            llm_client=llm_client,
            llm_model=llm_model,
        )
    return result["manifest"]
