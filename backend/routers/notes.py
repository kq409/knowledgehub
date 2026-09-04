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
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import db
from db import get_session
from models import Note, NoteChunk, NoteSourceType, ProcessingStatus, ReviewStatus
from schemas import (
    NoteChunkResponse,
    NoteCreate,
    NoteResponse,
    NoteStatus,
    NoteUpdate,
)
from schemas import (
    NoteSourceType as NoteSourceTypeSchema,
)
from services.note_parser import NoteParseError, NotePdfParser
from services.note_pipeline import (
    NotePipeline,
    apply_pdf_parse_result,
    apply_text_chunks,
    chunk_count_for,
    chunks_from_extracted_text,
    chunks_from_voice_note,
    mark_note_status,
    note_file_path,
)

router = APIRouter(prefix="/api/notes", tags=["notes"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
MAX_PDF_BYTES = 50 * 1024 * 1024

VOICE_REEMBED_FIELDS = {
    "title",
    "summary",
    "observations",
    "hypotheses",
    "questions",
    "next_steps",
    "cleaned_transcript",
}
HANDWRITTEN_REEMBED_FIELDS = {"extracted_text"}


def to_response(note: Note, chunk_count: int) -> NoteResponse:
    return NoteResponse(
        id=note.id,
        source_type=NoteSourceTypeSchema(note.source_type),
        title=note.title,
        summary=note.summary,
        observations=note.observations,
        hypotheses=note.hypotheses,
        questions=note.questions,
        next_steps=note.next_steps,
        tags=note.tags,
        raw_transcript=note.raw_transcript,
        cleaned_transcript=note.cleaned_transcript,
        review_status=ReviewStatus(note.review_status),
        model=note.model,
        prompt_version=note.prompt_version,
        original_filename=note.original_filename,
        page_count=note.page_count,
        extracted_text=note.extracted_text,
        paper_id=note.paper_id,
        processing_status=NoteStatus(note.processing_status),
        processing_error=note.processing_error,
        chunk_count=chunk_count,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


def build_note_pipeline(embeddings) -> NotePipeline:
    return NotePipeline(parser=NotePdfParser(), embeddings=embeddings)


def schedule_pipeline_warmup(app: FastAPI) -> None:
    if getattr(app.state, "note_pipeline", None) is not None:
        return
    embeddings = getattr(app.state, "embeddings", None)
    if embeddings is None:
        return
    app.state.note_pipeline = build_note_pipeline(embeddings)
    print("✅ Note pipeline ready", flush=True)


async def ensure_pipeline(app: FastAPI) -> NotePipeline:
    pipeline = getattr(app.state, "note_pipeline", None)
    if pipeline is not None:
        return pipeline

    embeddings = getattr(app.state, "embeddings", None)
    if embeddings is None:
        raise RuntimeError("Embedding service is not initialized")

    lock = getattr(app.state, "note_pipeline_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state.note_pipeline_lock = lock

    async with lock:
        pipeline = getattr(app.state, "note_pipeline", None)
        if pipeline is not None:
            return pipeline
        pipeline = build_note_pipeline(embeddings)
        app.state.note_pipeline = pipeline
        print("✅ Note pipeline ready", flush=True)
        return pipeline


def schedule_note_processing(app: FastAPI, note_id: uuid.UUID) -> None:
    jobs: set[asyncio.Task] = getattr(app.state, "note_jobs", None)
    if jobs is None:
        jobs = set()
        app.state.note_jobs = jobs
    task = asyncio.create_task(process_note(note_id, app))
    jobs.add(task)
    task.add_done_callback(jobs.discard)


async def process_note(note_id: uuid.UUID, app: FastAPI) -> None:
    if db.SessionLocal is None:
        print("❌ Note processing skipped: database is not initialized", flush=True)
        return

    print(f"⏳ Processing note {note_id}...", flush=True)
    async with db.SessionLocal() as session:
        await mark_note_status(session, note_id, ProcessingStatus.processing)
        note = await session.get(Note, note_id)
        if note is None:
            return
        source_type = note.source_type
        pdf_path = note.original_file
        fallback_title = (
            Path(note.original_filename).stem
            if note.original_filename
            else note.title or "Untitled note"
        )
        extracted_text = note.extracted_text

    try:
        pipeline = await ensure_pipeline(app)
        if source_type == NoteSourceType.handwritten.value:
            if extracted_text:
                chunks = chunks_from_extracted_text(extracted_text)
                embeddings = await asyncio.to_thread(pipeline.embed_chunks, chunks)
                async with db.SessionLocal() as session:
                    note = await session.get(Note, note_id)
                    if note is None:
                        return
                    await apply_text_chunks(session, note, chunks, embeddings)
            else:
                if not pdf_path:
                    raise NoteParseError("Original PDF path is missing")
                parsed = await asyncio.to_thread(
                    pipeline.parse_pdf, pdf_path, fallback_title
                )
                embeddings = await asyncio.to_thread(
                    pipeline.embed_chunks, parsed.chunks
                )
                async with db.SessionLocal() as session:
                    note = await session.get(Note, note_id)
                    if note is None:
                        return
                    await apply_pdf_parse_result(session, note, parsed, embeddings)
            print(f"✅ Note processed: {note_id}", flush=True)
            return

        async with db.SessionLocal() as session:
            note = await session.get(Note, note_id)
            if note is None:
                return
            chunks = chunks_from_voice_note(note)
            embeddings = await asyncio.to_thread(pipeline.embed_chunks, chunks)
            await apply_text_chunks(session, note, chunks, embeddings)
        print(f"✅ Note processed: {note_id}", flush=True)
    except Exception as exc:
        print(f"❌ Note processing failed: {exc}", flush=True)
        async with db.SessionLocal() as session:
            await mark_note_status(
                session, note_id, ProcessingStatus.failed, error=str(exc)[:2000]
            )


async def resume_pending_notes(app: FastAPI) -> None:
    if db.SessionLocal is None:
        return
    async with db.SessionLocal() as session:
        queued = await session.execute(
            select(Note.id).where(
                Note.processing_status.in_(
                    [
                        ProcessingStatus.pending.value,
                        ProcessingStatus.processing.value,
                    ]
                )
            )
        )
        ids = list(queued.scalars().all())
        missing_chunks = await session.execute(
            select(Note.id)
            .outerjoin(NoteChunk, Note.id == NoteChunk.note_id)
            .where(Note.processing_status == ProcessingStatus.ready.value)
            .group_by(Note.id)
            .having(func.count(NoteChunk.id) == 0)
        )
        for note_id in missing_chunks.scalars().all():
            if note_id not in ids:
                ids.append(note_id)
    if not ids:
        print("📝 No queued notes to process", flush=True)
        return
    print(f"⏳ Resuming {len(ids)} queued note(s)...", flush=True)
    for note_id in ids:
        await process_note(note_id, app)


def _should_reembed(note: Note, updates: dict) -> bool:
    if note.source_type == NoteSourceType.handwritten.value:
        return bool(HANDWRITTEN_REEMBED_FIELDS.intersection(updates))
    return bool(VOICE_REEMBED_FIELDS.intersection(updates))


@router.post("", response_model=NoteResponse, status_code=201)
@router.post("/", response_model=NoteResponse, status_code=201, include_in_schema=False)
async def create_note(
    request: Request,
    session: SessionDep,
    payload: NoteCreate,
):
    note = Note(
        source_type=NoteSourceType.voice.value,
        title=payload.title,
        summary=payload.summary,
        observations=payload.observations,
        hypotheses=payload.hypotheses,
        questions=payload.questions,
        next_steps=payload.next_steps,
        tags=payload.tags,
        raw_transcript=payload.raw_transcript,
        cleaned_transcript=payload.cleaned_transcript,
        review_status=payload.review_status.value,
        model=payload.model,
        prompt_version=payload.prompt_version,
        processing_status=ProcessingStatus.pending.value,
    )
    session.add(note)
    await session.commit()
    await session.refresh(note)
    print(f"⏳ Queued voice note for embedding: {note.title} ({note.id})", flush=True)
    schedule_note_processing(request.app, note.id)
    return to_response(note, 0)


@router.post("/upload", response_model=NoteResponse, status_code=202)
async def upload_note(
    request: Request,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
):
    filename = file.filename or "note.pdf"
    suffix = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    if suffix != ".pdf" and "pdf" not in content_type:
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail="PDF is larger than 50MB")

    note_id = uuid.uuid4()
    dest = note_file_path(note_id)
    dest.write_bytes(content)

    note = Note(
        id=note_id,
        source_type=NoteSourceType.handwritten.value,
        title=Path(filename).stem or "Untitled note",
        summary="",
        observations=[],
        hypotheses=[],
        questions=[],
        next_steps=[],
        tags=[],
        raw_transcript="",
        cleaned_transcript="",
        original_filename=filename,
        original_file=str(dest),
        review_status=ReviewStatus.accepted.value,
        processing_status=ProcessingStatus.pending.value,
    )
    session.add(note)
    await session.commit()
    await session.refresh(note)

    print(f"⏳ Queued PDF note for processing: {note.title} ({note.id})", flush=True)
    schedule_note_processing(request.app, note.id)
    return to_response(note, 0)


@router.get("", response_model=list[NoteResponse])
@router.get("/", response_model=list[NoteResponse], include_in_schema=False)
async def list_notes(
    session: SessionDep,
    source_type: Annotated[NoteSourceTypeSchema | None, Query()] = None,
):
    chunk_counts = (
        select(
            NoteChunk.note_id,
            func.count(NoteChunk.id).label("chunk_count"),
        )
        .group_by(NoteChunk.note_id)
        .subquery()
    )
    query = (
        select(Note, func.coalesce(chunk_counts.c.chunk_count, 0))
        .outerjoin(chunk_counts, Note.id == chunk_counts.c.note_id)
        .order_by(Note.created_at.desc())
    )
    if source_type is not None:
        query = query.where(Note.source_type == source_type.value)
    result = await session.execute(query)
    return [to_response(note, int(count)) for note, count in result.all()]


@router.get("/{note_id}", response_model=NoteResponse)
async def get_note(note_id: uuid.UUID, session: SessionDep):
    note = await session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return to_response(note, await chunk_count_for(session, note.id))


@router.get("/{note_id}/chunks", response_model=list[NoteChunkResponse])
async def list_note_chunks(note_id: uuid.UUID, session: SessionDep):
    note = await session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    result = await session.execute(
        select(NoteChunk)
        .where(NoteChunk.note_id == note_id)
        .order_by(NoteChunk.chunk_index.asc())
    )
    return list(result.scalars().all())


@router.get("/{note_id}/file")
async def download_note_file(note_id: uuid.UUID, session: SessionDep):
    note = await session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    if not note.original_file:
        raise HTTPException(status_code=404, detail="Original PDF is missing")
    path = Path(note.original_file)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Original PDF is missing")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=note.original_filename or f"{note.id}.pdf",
    )


@router.patch("/{note_id}", response_model=NoteResponse)
async def update_note(
    request: Request,
    note_id: uuid.UUID,
    payload: NoteUpdate,
    session: SessionDep,
):
    note = await session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")

    updates = payload.model_dump(exclude_unset=True)
    if "review_status" in updates and updates["review_status"] is not None:
        updates["review_status"] = updates["review_status"].value

    reembed = _should_reembed(note, updates)
    for field, value in updates.items():
        setattr(note, field, value)
    if reembed:
        note.processing_status = ProcessingStatus.pending.value
        note.processing_error = None
    note.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(note)
    if reembed:
        schedule_note_processing(request.app, note.id)
    return to_response(note, await chunk_count_for(session, note.id))


@router.delete("/{note_id}", status_code=204)
async def delete_note(note_id: uuid.UUID, session: SessionDep):
    note = await session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    path = Path(note.original_file) if note.original_file else None
    await session.delete(note)
    await session.commit()
    if path is not None and path.exists():
        path.unlink()
