import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from db import close_db, init_db, run_migrations
from routers.ask import router as ask_router
from routers.chat import router as chat_router
from routers.compare import router as compare_router
from routers.library import router as library_router
from routers.memories import router as memories_router
from routers.notes import resume_pending_notes
from routers.notes import router as notes_router
from routers.notes import schedule_pipeline_warmup as schedule_note_pipeline_warmup
from routers.papers import resume_pending_papers, schedule_pipeline_warmup
from routers.papers import router as papers_router
from routers.voice_notes import router as voice_notes_router
from services.agent.gate import EvidenceGate
from services.agent.loop import ResearchAgent
from services.ask import AskService
from services.compare import COMPARE_MAX_TOKENS, CompareService, max_tokens_from_env
from services.connect import (
    CONNECT_MAX_TOKENS,
    ConnectService,
    connect_min_similarity_from_env,
)
from services.embeddings import EmbeddingService
from services.extraction import ExtractionService
from services.retrieval import min_similarity_from_env
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
    print("🚀 Starting ResearchPilot...")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    print("🗄️  Running database migrations...")
    await asyncio.to_thread(run_migrations)
    init_db(database_url)
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
    evidence_gate = EvidenceGate(
        llm_client=llm_client,
        llm_model=llm_model,
        base_url=llm_base_url,
    )
    print(
        "🔍 Evidence gate: deterministic checks"
        + (" + LLM reviewer" if evidence_gate.uses_judge() else " only")
    )
    app.state.agent = ResearchAgent(
        llm_client=llm_client,
        llm_model=llm_model,
        embeddings=app.state.embeddings,
        gate=evidence_gate,
        compare=agent_compare,
        extraction=app.state.extraction,
        connect=app.state.connect,
    )
    app.state.paper_pipeline = None
    app.state.paper_pipeline_lock = asyncio.Lock()
    app.state.note_pipeline = None
    app.state.note_pipeline_lock = asyncio.Lock()
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
    print("✅ Ready!")

    async def resume_queued() -> None:
        try:
            await resume_pending_papers(app)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"⚠️  Failed to resume queued papers: {exc}", flush=True)
        try:
            await resume_pending_notes(app)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"⚠️  Failed to resume queued notes: {exc}", flush=True)

    resume_task = asyncio.create_task(resume_queued())
    yield
    resume_task.cancel()
    warmup = getattr(app.state, "paper_pipeline_warmup", None)
    if warmup is not None:
        warmup.cancel()
    await close_db()


app = FastAPI(
    title="ResearchPilot",
    description="Personal research assistant: voice notes, paper library, retrieval, and comparison",
    lifespan=lifespan,
)

# CORS for localhost development (Vite on 3000, docker-compose publishes 8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

app.include_router(voice_notes_router)
app.include_router(notes_router)
app.include_router(papers_router)
app.include_router(library_router)
app.include_router(ask_router)
app.include_router(chat_router)
app.include_router(compare_router)
app.include_router(memories_router)


@app.get("/api/status")
async def get_status():
    return {
        "status": "ready" if service else "initializing",
        "whisper_model": os.getenv("WHISPER_MODEL"),
        "llm_model": os.getenv("LLM_MODEL"),
        "llm_base_url": os.getenv("LLM_BASE_URL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
    }


@app.get("/api/system-prompt")
async def get_system_prompt():
    if not service:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {"default_prompt": service.get_default_system_prompt()}


@app.post("/api/transcribe")
async def transcribe_audio(audio: Annotated[UploadFile, File()]):
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
