from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from routers.health import router as health_router


def _app() -> FastAPI:
    application = FastAPI()
    application.include_router(health_router)
    return application


async def test_healthz_does_not_need_a_database():
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_status_hides_llm_url_in_demo(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("LLM_BASE_URL", "http://secret-llm.internal/v1")
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["demo_mode"] is True
    assert "llm_base_url" not in body
    assert body["replicas"] == 1


async def test_status_includes_llm_url_locally(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434/v1")
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["llm_base_url"] == "http://ollama:11434/v1"
