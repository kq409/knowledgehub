from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from eval.store import write_run
from routers.eval import router as eval_router


async def test_eval_runs_list_and_detail(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.store.RUNS_DIR", tmp_path)
    write_run(
        "20260101T000000Z",
        {"suite": "ask", "summary": {"pass_rate": 1.0, "tasks": 1}},
        [
            {
                "task_id": "acronym-ella",
                "trial_index": 0,
                "query": "ZXQELLA7",
                "passed": True,
                "score": 1.0,
                "grades": [],
            }
        ],
    )
    app = FastAPI()
    app.include_router(eval_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/api/eval/runs")
        assert listed.status_code == 200
        body = listed.json()
        assert body[0]["id"] == "20260101T000000Z"
        detail = await client.get("/api/eval/runs/20260101T000000Z")
        assert detail.status_code == 200
        assert detail.json()["trials"][0]["task_id"] == "acronym-ella"
        missing = await client.get("/api/eval/runs/nope")
        assert missing.status_code == 404
