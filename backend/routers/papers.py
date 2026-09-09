import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import db
from db import get_session
from models import DigestStatus, Paper, PaperChunk, PaperStatus
from routers.notes import to_response_with_links as note_to_response
from schemas import (
    DigestStatus as DigestStatusSchema,
)
from schemas import (
    NoteResponse,
    PaperChunkResponse,
    PaperResponse,
    PaperUpdate,
)
from services.agent.notifications import FAILED, notify_ingested
from services.library_ingest import create_pending_paper, validate_pdf_bytes
from services.note_links import links_by_note_ids, notes_linked_to_paper
from services.note_pipeline import chunk_count_for as note_chunk_count_for
from services.paper_digest import generate_and_store_digest, paper_digest_from
from services.paper_parser import PaperParser
from services.paper_pipeline import (
    PaperPipeline,
    apply_parse_result,
    chunk_count_for,
    mark_paper_status,
)

router = APIRouter(prefix="/api/papers", tags=["papers"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def to_response(paper: Paper, chunk_count: int) -> PaperResponse:
    digest_status = paper.digest_status or DigestStatus.pending.value
    try:
        parsed_digest_status = DigestStatusSchema(digest_status)
    except ValueError:
        parsed_digest_status = DigestStatusSchema.pending
    return PaperResponse(
        id=paper.id,
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        abstract=paper.abstract,
        summary=paper.summary,
        digest=paper_digest_from(paper.digest),
        digest_status=parsed_digest_status,
        source=paper.source,
        tags=paper.tags,
        original_filename=paper.original_filename,
        page_count=paper.page_count,
        processing_status=PaperStatus(paper.processing_status),
        processing_error=paper.processing_error,
        chunk_count=chunk_count,
        created_at=paper.created_at,
        updated_at=paper.updated_at,
    )


def build_paper_pipeline(embeddings) -> PaperPipeline:
    parser = PaperParser()
    parser.warmup()
    return PaperPipeline(parser=parser, embeddings=embeddings)


def schedule_pipeline_warmup(app: FastAPI) -> None:
    """Ping GROBID in the background after the API is up."""
    if getattr(app.state, "paper_pipeline", None) is not None:
        return
    if getattr(app.state, "paper_pipeline_warmup", None) is not None:
        return
    print("⏳ Warming up GROBID paper parser...", flush=True)
    app.state.paper_pipeline_warmup = asyncio.create_task(
        asyncio.to_thread(build_paper_pipeline, app.state.embeddings)
    )


async def ensure_pipeline(app: FastAPI) -> PaperPipeline:
    pipeline = getattr(app.state, "paper_pipeline", None)
    if pipeline is not None:
        return pipeline

    lock = getattr(app.state, "paper_pipeline_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state.paper_pipeline_lock = lock

    async with lock:
        pipeline = getattr(app.state, "paper_pipeline", None)
        if pipeline is not None:
            return pipeline

        warmup = getattr(app.state, "paper_pipeline_warmup", None)
        if warmup is None:
            print("⏳ Initializing paper pipeline...", flush=True)
            warmup = asyncio.create_task(
                asyncio.to_thread(build_paper_pipeline, app.state.embeddings)
            )
            app.state.paper_pipeline_warmup = warmup

        try:
            pipeline = await warmup
        except Exception:
            app.state.paper_pipeline_warmup = None
            raise

        app.state.paper_pipeline = pipeline
        print("✅ Paper pipeline ready", flush=True)
        return pipeline


def schedule_paper_processing(app: FastAPI, paper_id: uuid.UUID) -> None:
    jobs: set[asyncio.Task] = getattr(app.state, "paper_jobs", None)
    if jobs is None:
        jobs = set()
        app.state.paper_jobs = jobs
    task = asyncio.create_task(process_uploaded_paper(paper_id, app))
    jobs.add(task)
    task.add_done_callback(jobs.discard)


async def process_uploaded_paper(paper_id: uuid.UUID, app: FastAPI) -> None:
    if db.SessionLocal is None:
        print("❌ Paper processing skipped: database is not initialized", flush=True)
        return

    print(f"⏳ Processing paper {paper_id}...", flush=True)
    async with db.SessionLocal() as session:
        await mark_paper_status(session, paper_id, PaperStatus.processing)
        paper = await session.get(Paper, paper_id)
        if paper is None:
            return
        pdf_path = paper.original_file
        fallback_title = Path(paper.original_filename).stem or "Untitled paper"

    try:
        pipeline = await ensure_pipeline(app)
        parsed, embeddings = await asyncio.to_thread(
            _run_parse_and_embed, pipeline, pdf_path, fallback_title
        )
        async with db.SessionLocal() as session:
            paper = await session.get(Paper, paper_id)
            if paper is None:
                return
            await apply_parse_result(session, paper, parsed, embeddings)
            print(
                f"✅ Paper processed: {paper.title} ({len(parsed.chunks)} chunks)",
                flush=True,
            )
            notify_ingested("paper", paper.title, detail=f"{len(parsed.chunks)} chunks")
        await _digest_processed_paper(app, paper_id)
    except Exception as exc:
        print(f"❌ Paper processing failed: {exc}", flush=True)
        notify_ingested("paper", fallback_title, outcome=FAILED, detail=str(exc)[:200])
        async with db.SessionLocal() as session:
            await mark_paper_status(
                session, paper_id, PaperStatus.failed, error=str(exc)[:2000]
            )


async def resume_pending_papers(app: FastAPI) -> None:
    if db.SessionLocal is None:
        return
    async with db.SessionLocal() as session:
        result = await session.execute(
            select(Paper.id).where(
                Paper.processing_status.in_(
                    [PaperStatus.pending.value, PaperStatus.processing.value]
                )
            )
        )
        ids = list(result.scalars().all())
    if not ids:
        print("📄 No queued papers to process", flush=True)
        return
    print(f"⏳ Resuming {len(ids)} queued paper(s)...", flush=True)
    for paper_id in ids:
        await process_uploaded_paper(paper_id, app)


async def resume_missing_digests(app: FastAPI) -> None:
    """Generate stored understandings for ready papers that never got one."""
    if db.SessionLocal is None:
        return
    async with db.SessionLocal() as session:
        result = await session.execute(
            select(Paper.id).where(
                Paper.processing_status == PaperStatus.ready.value,
                Paper.digest_status != DigestStatus.ready.value,
            )
        )
        ids = list(result.scalars().all())
    if not ids:
        return
    print(f"⏳ Generating digest for {len(ids)} paper(s)...", flush=True)
    for paper_id in ids:
        await _digest_processed_paper(app, paper_id)


async def _digest_processed_paper(app: FastAPI, paper_id: uuid.UUID) -> None:
    agent = getattr(app.state, "agent", None)
    llm_client = getattr(agent, "llm_client", None)
    llm_model = getattr(agent, "llm_model", None)
    if llm_client is None or not llm_model or db.SessionLocal is None:
        return
    try:
        async with db.SessionLocal() as session:
            paper = await session.get(Paper, paper_id)
            if paper is None:
                return
            if paper.processing_status != PaperStatus.ready.value:
                return
            await generate_and_store_digest(
                session,
                paper,
                llm_client=llm_client,
                llm_model=llm_model,
            )
            print(f"🧠 Paper digest {paper.digest_status}: {paper.title}", flush=True)
    except Exception as exc:
        print(f"⚠️  Paper digest failed for {paper_id}: {exc}", flush=True)


def _run_parse_and_embed(pipeline: PaperPipeline, pdf_path: str, fallback_title: str):
    parsed = pipeline.parse_and_embed(pdf_path, fallback_title)
    embeddings = pipeline.embed_chunks(parsed)
    return parsed, embeddings


@router.post("", response_model=PaperResponse, status_code=202)
@router.post(
    "/", response_model=PaperResponse, status_code=202, include_in_schema=False
)
async def upload_paper(
    request: Request,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
):
    content = await file.read()
    try:
        filename = validate_pdf_bytes(
            file.filename,
            file.content_type,
            content,
            fallback_name="paper.pdf",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    paper = await create_pending_paper(session, content, filename)
    print(f"⏳ Queued paper for processing: {paper.title} ({paper.id})", flush=True)
    schedule_paper_processing(request.app, paper.id)
    return to_response(paper, 0)


@router.get("", response_model=list[PaperResponse])
@router.get("/", response_model=list[PaperResponse], include_in_schema=False)
async def list_papers(session: SessionDep):
    chunk_counts = (
        select(
            PaperChunk.paper_id,
            func.count(PaperChunk.id).label("chunk_count"),
        )
        .group_by(PaperChunk.paper_id)
        .subquery()
    )
    result = await session.execute(
        select(Paper, func.coalesce(chunk_counts.c.chunk_count, 0))
        .outerjoin(chunk_counts, Paper.id == chunk_counts.c.paper_id)
        .order_by(Paper.created_at.desc())
    )
    return [to_response(paper, int(count)) for paper, count in result.all()]


@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: uuid.UUID, session: SessionDep):
    paper = await session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return to_response(paper, await chunk_count_for(session, paper.id))


@router.get("/{paper_id}/chunks", response_model=list[PaperChunkResponse])
async def list_paper_chunks(paper_id: uuid.UUID, session: SessionDep):
    paper = await session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    result = await session.execute(
        select(PaperChunk)
        .where(PaperChunk.paper_id == paper_id)
        .order_by(PaperChunk.chunk_index.asc())
    )
    return list(result.scalars().all())


@router.get("/{paper_id}/notes", response_model=list[NoteResponse])
async def list_paper_notes(paper_id: uuid.UUID, session: SessionDep):
    paper = await session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    notes = await notes_linked_to_paper(session, paper_id)
    links_map = await links_by_note_ids(session, [note.id for note in notes])
    responses: list[NoteResponse] = []
    for note in notes:
        count = await note_chunk_count_for(session, note.id)
        responses.append(note_to_response(note, count, links_map.get(note.id, [])))
    return responses


@router.get("/{paper_id}/file")
async def download_paper_file(paper_id: uuid.UUID, session: SessionDep):
    paper = await session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    path = Path(paper.original_file)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Original PDF is missing")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=paper.original_filename,
    )


@router.patch("/{paper_id}", response_model=PaperResponse)
async def update_paper(paper_id: uuid.UUID, payload: PaperUpdate, session: SessionDep):
    paper = await session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(paper, field, value)
    paper.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(paper)
    return to_response(paper, await chunk_count_for(session, paper.id))


@router.delete("/{paper_id}", status_code=204)
async def delete_paper(paper_id: uuid.UUID, session: SessionDep):
    paper = await session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    path = Path(paper.original_file)
    await session.delete(paper)
    await session.commit()
    if path.exists():
        path.unlink()
