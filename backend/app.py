import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

import db
from routers.ask import router as ask_router
from routers.chat import router as chat_router
from routers.compare import router as compare_router
from routers.conversations import router as conversations_router
from routers.demo import router as demo_router
from routers.documents import resume_pending_documents
from routers.documents import router as documents_router
from routers.eval import router as eval_router
from routers.health import router as health_router
from routers.library import router as library_router
from routers.memories import router as memories_router
from routers.notes import resume_pending_notes
from routers.notes import router as notes_router
from routers.notes import schedule_pipeline_warmup as schedule_note_pipeline_warmup
from routers.papers import (
    resume_missing_digests,
    resume_pending_papers,
    schedule_pipeline_warmup,
)
from routers.papers import router as papers_router
from routers.records import router as records_router
from routers.voice_notes import router as voice_notes_router
from services.agent.approvals import ApprovalBroker
from services.agent.gate import EvidenceGate
from services.agent.loop import ResearchAgent
from services.agent.mcp_client import McpConfigError, McpRegistry, load_server_configs
from services.agent.permissions import approval_enabled
from services.agent.skills import reload_skills
from services.agent.telemetry import configure_agent_logging
from services.ask import AskService
from services.compare import COMPARE_MAX_TOKENS, CompareService, max_tokens_from_env
from services.connect import (
    CONNECT_MAX_TOKENS,
    ConnectService,
    connect_min_similarity_from_env,
)
from services.demo import seed_demo_library
from services.document_pipeline import DocumentPipeline
from services.embeddings import EmbeddingService
from services.extraction import ExtractionService
from services.identity import IdentityDep
from services.retrieval import min_similarity_from_env
from services.settings import (
    cors_origin_regex,
    cors_origins,
    demo_mode,
    frontend_dist,
    grobid_enabled,
    whisper_enabled,
)
from transcription import TranscriptionService

load_dotenv()


class CleanRequest(BaseModel):
    text: str
    system_prompt: str | None = None


service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uses OpenAI-compatible API (Ollama, OpenAI, LM Studio, etc.). Configure via .env file."""
    global service
    print("🚀 Starting KnowledgeHub...")
    configure_agent_logging()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    print("🗄️  Running database migrations...")
    await asyncio.to_thread(db.run_migrations)
    db.init_db(database_url)
    print("✅ Database connected")

    llm_base_url = os.getenv("LLM_BASE_URL")
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_model = os.getenv("LLM_MODEL")
    print(f"🔄 Connecting to LLM at {llm_base_url}...")
    llm_client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
    try:
        await asyncio.wait_for(asyncio.to_thread(llm_client.models.list), timeout=5)
        print("✅ Connected to LLM API!")
    except Exception as exc:
        print(f"⚠️  Warning: Could not connect to LLM: {exc}")
        print(f"   Make sure your LLM server is running at {llm_base_url}")

    # Embeddings often stay on local Ollama (nomic-embed-text) even when the
    # chat model is a cloud API that has no embedding endpoint.
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or llm_base_url
    embedding_api_key = os.getenv("EMBEDDING_API_KEY") or llm_api_key
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    if embedding_base_url.rstrip("/") != (llm_base_url or "").rstrip("/"):
        print(f"🔄 Embeddings via {embedding_base_url} ({embedding_model})...")
        embedding_client = OpenAI(
            base_url=embedding_base_url, api_key=embedding_api_key
        )
    else:
        embedding_client = llm_client

    app.state.extraction = ExtractionService(
        llm_client=llm_client,
        llm_model=llm_model,
    )
    app.state.embeddings = EmbeddingService(
        llm_client=embedding_client,
        model=embedding_model,
    )
    app.state.ask = AskService(
        llm_client=llm_client,
        llm_model=llm_model,
        embeddings=app.state.embeddings,
        min_similarity=min_similarity_from_env(),
    )
    compare_max_tokens = max_tokens_from_env(
        "COMPARE_MAX_TOKENS", fallback=COMPARE_MAX_TOKENS
    )
    app.state.compare = CompareService(
        llm_client=llm_client,
        llm_model=llm_model,
        embeddings=app.state.embeddings,
        max_tokens=compare_max_tokens,
    )
    app.state.connect = ConnectService(
        llm_client=llm_client,
        llm_model=llm_model,
        embeddings=app.state.embeddings,
        min_similarity=connect_min_similarity_from_env(),
        max_tokens=max_tokens_from_env("CONNECT_MAX_TOKENS", CONNECT_MAX_TOKENS),
    )
    # The agent runs comparisons inside a chat turn, where a slow model hurts
    # more than it does on the Compare tab, so it gets its own ceiling.
    agent_compare = CompareService(
        llm_client=llm_client,
        llm_model=llm_model,
        embeddings=app.state.embeddings,
        max_tokens=max_tokens_from_env(
            "AGENT_COMPARE_MAX_TOKENS", fallback=compare_max_tokens
        ),
    )
    # Scanned once here so the catalog is warm and its cost is paid at startup
    # rather than inside the first chat request.
    installed_skills = reload_skills()
    print(f"📚 Skills: {len(installed_skills)} installed")
    evidence_gate = EvidenceGate(
        llm_client=llm_client,
        llm_model=llm_model,
        base_url=llm_base_url,
    )
    print(
        "🔍 Evidence gate: deterministic checks"
        + (" + LLM reviewer" if evidence_gate.uses_judge() else " only")
    )
    mcp_registry = None
    try:
        mcp_configs = [item for item in load_server_configs() if item.enabled]
    except McpConfigError as exc:
        print(f"⚠️  MCP config skipped: {exc}")
        mcp_configs = []
    if mcp_configs:
        mcp_registry = McpRegistry(mcp_configs)
        await mcp_registry.connect()
        print(f"🔌 MCP: {len(mcp_registry.tools())} external tool(s)")
    approver = ApprovalBroker() if approval_enabled() else None
    if approver is not None:
        print("✍️  Write tools wait for approval")
    app.state.approvals = approver
    app.state.mcp = mcp_registry
    app.state.agent = ResearchAgent(
        llm_client=llm_client,
        llm_model=llm_model,
        embeddings=app.state.embeddings,
        gate=evidence_gate,
        compare=agent_compare,
        extraction=app.state.extraction,
        connect=app.state.connect,
        approver=approver,
        mcp=mcp_registry,
    )
    app.state.paper_pipeline = None
    app.state.paper_pipeline_lock = asyncio.Lock()
    app.state.note_pipeline = None
    app.state.note_pipeline_lock = asyncio.Lock()
    app.state.document_pipeline = DocumentPipeline(app.state.embeddings)
    schedule_pipeline_warmup(app)
    schedule_note_pipeline_warmup(app)

    # Whisper stays lazy so the API comes up before speech models load.
    service = TranscriptionService(
        whisper_model=os.getenv("WHISPER_MODEL"),
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_client=llm_client,
        load_whisper=False,
    )
    if not whisper_enabled():
        print("🔇 Whisper disabled (demo/prod)")
    if not grobid_enabled():
        print("📄 GROBID disabled; papers parse with pypdf")
    print("✅ Ready!")

    if demo_mode() and db.SessionLocal is not None:
        try:
            async with db.SessionLocal() as session:
                await seed_demo_library(session, app.state.embeddings)
        except Exception as exc:
            print(f"⚠️  Demo seed failed: {exc}", flush=True)

    async def resume_queued() -> None:
        try:
            await resume_pending_papers(app)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"⚠️  Failed to resume queued papers: {exc}", flush=True)
        try:
            await resume_missing_digests(app)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"⚠️  Failed to resume paper digests: {exc}", flush=True)
        try:
            await resume_pending_notes(app)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"⚠️  Failed to resume queued notes: {exc}", flush=True)
        try:
            await resume_pending_documents(app)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"⚠️  Failed to resume queued documents: {exc}", flush=True)

    resume_task = asyncio.create_task(resume_queued())
    yield
    resume_task.cancel()
    warmup = getattr(app.state, "paper_pipeline_warmup", None)
    if warmup is not None:
        warmup.cancel()
    mcp = getattr(app.state, "mcp", None)
    if mcp is not None:
        await mcp.close()
    await db.close_db()


app = FastAPI(
    title="KnowledgeHub",
    description="Knowledge base for notes, papers, retrieval, and comparison",
    lifespan=lifespan,
)

# Same-origin in demo/prod (FastAPI serves frontend/dist). Localhost regex
# stays for Vite. Set CORS_ORIGINS to a real origin if the UI is hosted apart.
_cors_regex = cors_origin_regex()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_origin_regex=_cors_regex,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

app.include_router(health_router)
app.include_router(demo_router)
app.include_router(voice_notes_router)
app.include_router(notes_router)
app.include_router(papers_router)
app.include_router(documents_router)
app.include_router(library_router)
app.include_router(records_router)
app.include_router(ask_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(compare_router)
app.include_router(memories_router)
app.include_router(eval_router)


@app.get("/api/identity")
async def get_current_identity(
    identity: IdentityDep,
    session: Annotated[AsyncSession, Depends(db.get_session)],
):
    from services.library_records import list_visible_spaces

    spaces = await list_visible_spaces(session, identity)
    return {
        "user_id": identity.user_id,
        "space_ids": [str(item) for item in sorted(identity.space_ids, key=str)],
        "primary_space_id": str(identity.primary_space_id),
        "spaces": [
            {"id": str(row.id), "slug": row.slug, "name": row.name} for row in spaces
        ],
        "stub": True,
        "notice": "Demo ACL via X-User-Id header, not SSO.",
    }


@app.get("/api/system-prompt")
async def get_system_prompt():
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {"default_prompt": service.get_default_system_prompt()}


@app.post("/api/transcribe")
async def transcribe_audio(audio: Annotated[UploadFile, File()]):
    if not whisper_enabled():
        raise HTTPException(
            status_code=403,
            detail="Voice transcription is disabled on this deployment.",
        )
    if not service:
        raise HTTPException(
            status_code=503, detail="Service not ready, still initializing models"
        )

    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw_text = service.transcribe(tmp_path)
        return {"success": True, "text": raw_text}

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {str(e)}"
        ) from e

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/clean")
async def clean_text(request: CleanRequest):
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        stream = service.open_clean_stream(
            request.text, system_prompt=request.system_prompt
        )
    except Exception as e:
        # Log the full error to the backend terminal; keep the response generic so no
        # raw error detail leaks to the frontend.
        print(f"❌ LLM cleaning failed: {e}")
        raise HTTPException(
            status_code=502,
            detail="LLM cleaning failed. Check the backend terminal for details.",
        ) from e

    def generate():
        try:
            yield from service.iter_clean_tokens(stream)
        except Exception as e:
            print(f"❌ LLM cleaning failed: {e}")

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _mount_frontend(application: FastAPI) -> None:
    configured = frontend_dist()
    dist = Path(
        configured or Path(__file__).resolve().parent.parent / "frontend" / "dist"
    )
    if not dist.is_dir():
        return
    assets = dist / "assets"
    if assets.is_dir():
        application.mount("/assets", StaticFiles(directory=assets), name="assets")

    @application.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = dist / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


_mount_frontend(app)
