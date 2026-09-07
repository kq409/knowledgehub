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
from models import (
    Note,
    NoteChunk,
    NotePaper,
    NoteSourceType,
    ProcessingStatus,
    ReviewStatus,
)
from schemas import (
    ConnectReasonMode,
    ConnectRequest,
    ConnectResponse,
    NoteChunkResponse,
    NoteCreate,
    NoteLinkSource,
    NotePaperLink,
    NoteResponse,
    NoteStatus,
    NoteUpdate,
)
from schemas import (
    NoteSourceType as NoteSourceTypeSchema,
)
from services.connect import (
    ConnectError,
    ConnectNotFoundError,
    ConnectNotReadyError,
    ConnectService,
    empty_connect_response,
    response_from_stored,
)
from services.library_ingest import (
    create_pending_handwritten_note,
    validate_pdf_bytes,
)
from services.note_links import (
    NoteLinkError,
    links_by_note_ids,
    list_links_for_note,
    list_paper_ids_for_note,
    list_skipped_paper_ids,
    replace_note_papers,
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
)

router = APIRouter(prefix="/api/notes", tags=["notes"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_connect_service(request: Request) -> ConnectService:
    service = getattr(request.app.state, "connect", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Connect service not ready")
    return service


ConnectDep = Annotated[ConnectService, Depends(get_connect_service)]

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


def _parse_related_at(note: Note) -> datetime | None:
    stored = note.related_result
    if not isinstance(stored, dict):
        return None
    raw = stored.get("generated_at")
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def schema_links(rows: list[NotePaper] | None) -> list[NotePaperLink]:
    links: list[NotePaperLink] = []
    for row in rows or []:
        reason_mode = None
        if row.reason_mode:
            try:
                reason_mode = ConnectReasonMode(row.reason_mode)
            except ValueError:
                reason_mode = None
        try:
            source = NoteLinkSource(row.source)
        except ValueError:
            source = NoteLinkSource.researcher
        links.append(
            NotePaperLink(
                paper_id=row.paper_id,
                source=source,
                similarity=row.similarity,
                snippet=row.snippet,
                reason=row.reason,
                reason_mode=reason_mode,
            )
        )
    return links


def to_response(
    note: Note,
    chunk_count: int,
    paper_ids: list[uuid.UUID] | None = None,
    paper_links: list[NotePaperLink] | None = None,
) -> NoteResponse:
    links = list(paper_links or [])
    ids = (
        list(paper_ids) if paper_ids is not None else [item.paper_id for item in links]
    )
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
        paper_ids=ids,
        paper_links=links,
        related_generated_at=_parse_related_at(note),
        processing_status=NoteStatus(note.processing_status),
        processing_error=note.processing_error,
        chunk_count=chunk_count,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


def to_response_with_links(
    note: Note, chunk_count: int, rows: list[NotePaper] | None = None
) -> NoteResponse:
    links = schema_links(rows)
    return to_response(
        note,
        chunk_count,
        [item.paper_id for item in links],
        links,
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


def schedule_connect(app: FastAPI, note_id: uuid.UUID) -> None:
    if getattr(app.state, "connect", None) is None:
        return
    jobs: set[asyncio.Task] = getattr(app.state, "connect_jobs", None)
    if jobs is None:
        jobs = set()
        app.state.connect_jobs = jobs
    task = asyncio.create_task(_run_connect(note_id, app))
    jobs.add(task)
    task.add_done_callback(jobs.discard)


async def _store_empty_related(note_id: uuid.UUID, app: FastAPI) -> None:
    if db.SessionLocal is None:
        return
    try:
        async with db.SessionLocal() as session:
            note = await session.get(Note, note_id)
            if note is None or note.related_result:
                return
            linked = await list_paper_ids_for_note(session, note_id)
            skipped = await list_skipped_paper_ids(session, note_id)
            connect = getattr(app.state, "connect", None)
            model = connect.llm_model if connect is not None else None
            note.related_result = empty_connect_response(
                note_id,
                reason_mode=ConnectReasonMode.llm,
                linked_paper_ids=linked,
                skipped_paper_ids=list(skipped),
                model=model,
            ).model_dump(mode="json")
            await session.commit()
    except Exception as exc:
        print(
            f"⚠️  Could not store empty related result for {note_id}: {exc}",
            flush=True,
        )


async def _run_connect(note_id: uuid.UUID, app: FastAPI) -> None:
    connect = getattr(app.state, "connect", None)
    if connect is None or db.SessionLocal is None:
        return
    try:
        async with db.SessionLocal() as session:
            await connect.connect(session, note_id)
        print(f"🔗 Related papers linked for note {note_id}", flush=True)
    except ConnectNotReadyError:
        print(f"⚠️  Connect skipped; note {note_id} is not ready", flush=True)
    except ConnectNotFoundError:
        print(f"⚠️  Connect skipped; note {note_id} was deleted", flush=True)
    except Exception as exc:
        print(f"⚠️  Connect after note {note_id} failed: {exc}", flush=True)
        await _store_empty_related(note_id, app)


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
            schedule_connect(app, note_id)
            return

        async with db.SessionLocal() as session:
            note = await session.get(Note, note_id)
            if note is None:
                return
            chunks = chunks_from_voice_note(note)
            embeddings = await asyncio.to_thread(pipeline.embed_chunks, chunks)
            await apply_text_chunks(session, note, chunks, embeddings)
        print(f"✅ Note processed: {note_id}", flush=True)
        schedule_connect(app, note_id)
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
    await session.flush()
    try:
        await replace_note_papers(session, note.id, payload.paper_ids)
    except NoteLinkError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await session.commit()
    await session.refresh(note)
    print(f"⏳ Queued voice note for embedding: {note.title} ({note.id})", flush=True)
    schedule_note_processing(request.app, note.id)
    links = await list_links_for_note(session, note.id)
    return to_response_with_links(note, 0, links)


@router.post("/upload", response_model=NoteResponse, status_code=202)
async def upload_note(
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
            fallback_name="note.pdf",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    note = await create_pending_handwritten_note(session, content, filename)
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
    rows = result.all()
    links_map = await links_by_note_ids(session, [note.id for note, _ in rows])
    return [
        to_response_with_links(note, int(count), links_map.get(note.id, []))
        for note, count in rows
    ]


@router.get("/{note_id}", response_model=NoteResponse)
async def get_note(note_id: uuid.UUID, session: SessionDep):
    note = await session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return to_response_with_links(
        note,
        await chunk_count_for(session, note.id),
        await list_links_for_note(session, note.id),
    )


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


@router.post("/{note_id}/related", response_model=ConnectResponse)
async def connect_note_to_literature(
    note_id: uuid.UUID,
    session: SessionDep,
    connect_service: ConnectDep,
    payload: ConnectRequest | None = None,
):
    try:
        return await connect_service.connect(
            session, note_id, payload or ConnectRequest()
        )
    except ConnectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConnectNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConnectError as exc:
        print(f"❌ Connect generation failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Could not find related papers. Check the backend terminal for details.",
        ) from exc
    except Exception as exc:
        print(f"❌ Connect failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Connect failed. Check the backend terminal for details.",
        ) from exc


@router.get("/{note_id}/related", response_model=ConnectResponse)
async def get_related_papers(note_id: uuid.UUID, session: SessionDep):
    note = await session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    stored = response_from_stored(note)
    if stored is None:
        raise HTTPException(status_code=404, detail="No related-papers run stored yet")
    return stored


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
    paper_ids_update = updates.pop("paper_ids", None)

    reembed = _should_reembed(note, updates)
    for field, value in updates.items():
        setattr(note, field, value)
    if paper_ids_update is not None:
        try:
            await replace_note_papers(session, note.id, paper_ids_update)
        except NoteLinkError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if reembed:
        note.processing_status = ProcessingStatus.pending.value
        note.processing_error = None
    note.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(note)
    if reembed:
        schedule_note_processing(request.app, note.id)
    return to_response_with_links(
        note,
        await chunk_count_for(session, note.id),
        await list_links_for_note(session, note.id),
    )


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
