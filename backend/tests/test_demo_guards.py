from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from routers.eval import router as eval_router


async def test_eval_run_forbidden_in_demo(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("EVAL_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(eval_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/eval/run")
    assert response.status_code == 403


async def test_demo_reset_forbidden_without_token(monkeypatch):
    monkeypatch.setenv("DEMO_RESET_TOKEN", "reset-secret")
    from routers.demo import router as demo_router

    app = FastAPI()
    app.include_router(demo_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.post("/api/demo/reset")
        wrong = await client.post(
            "/api/demo/reset", headers={"X-Demo-Reset-Token": "nope"}
        )
    assert denied.status_code == 403
    assert wrong.status_code == 403
