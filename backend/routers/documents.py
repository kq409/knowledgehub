import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import db
from db import get_session
from models import LibraryDocument, LibraryDocumentChunk, ProcessingStatus
from schemas import LibraryDocumentResponse, LibraryDocumentUpdate, PaperStatus
from services.agent.notifications import FAILED, notify_ingested
from services.document_pipeline import (
    DocumentPipeline,
    apply_parse_result,
    chunk_count_for,
    mark_document_status,
)
from services.identity import IdentityDep, require_visible, space_clause

router = APIRouter(prefix="/api/documents", tags=["documents"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def to_response(document: LibraryDocument, chunk_count: int) -> LibraryDocumentResponse:
    return LibraryDocumentResponse(
        id=document.id,
        title=document.title,
        original_filename=document.original_filename,
        mime_type=document.mime_type,
        extracted_text=document.extracted_text,
        processing_status=PaperStatus(document.processing_status),
        processing_error=document.processing_error,
        chunk_count=chunk_count,
        revision=int(document.revision or 1),
        sha256=document.sha256,
        accession_status=document.accession_status,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def schedule_document_processing(app: FastAPI, document_id: uuid.UUID) -> None:
    """Parse/embed this document in-process. Production should use a queue."""
    jobs: set[asyncio.Task] = getattr(app.state, "document_jobs", None)
    if jobs is None:
        jobs = set()
        app.state.document_jobs = jobs
    task = asyncio.create_task(process_uploaded_document(document_id, app))
    jobs.add(task)
    task.add_done_callback(jobs.discard)


async def process_uploaded_document(document_id: uuid.UUID, app: FastAPI) -> None:
    if db.SessionLocal is None:
        print("❌ Document processing skipped: database is not initialized", flush=True)
        return

    print(f"⏳ Processing document {document_id}...", flush=True)
    async with db.SessionLocal() as session:
        await mark_document_status(session, document_id, ProcessingStatus.processing)
        document = await session.get(LibraryDocument, document_id)
        if document is None:
            return
        file_path = document.original_file
        fallback_title = Path(document.original_filename).stem or "Untitled document"

    try:
        pipeline = getattr(app.state, "document_pipeline", None)
        if pipeline is None:
            pipeline = DocumentPipeline(app.state.embeddings)
            app.state.document_pipeline = pipeline
        parsed, embeddings = await asyncio.to_thread(
            _run_parse_and_embed, pipeline, file_path, fallback_title
        )
        async with db.SessionLocal() as session:
            document = await session.get(LibraryDocument, document_id)
            if document is None:
                return
            await apply_parse_result(session, document, parsed, embeddings)
            print(
                f"✅ Document processed: {document.title} ({len(parsed.chunks)} chunks)",
                flush=True,
            )
            notify_ingested(
                "document", document.title, detail=f"{len(parsed.chunks)} chunks"
            )
    except Exception as exc:
        print(f"❌ Document processing failed: {exc}", flush=True)
        notify_ingested(
            "document", fallback_title, outcome=FAILED, detail=str(exc)[:200]
        )
        async with db.SessionLocal() as session:
            await mark_document_status(
                session,
                document_id,
                ProcessingStatus.failed,
                error=str(exc)[:2000],
            )


async def resume_pending_documents(app: FastAPI) -> None:
    if db.SessionLocal is None:
        return
    async with db.SessionLocal() as session:
        result = await session.execute(
            select(LibraryDocument.id).where(
                LibraryDocument.processing_status.in_(
                    [ProcessingStatus.pending.value, ProcessingStatus.processing.value]
                )
            )
        )
        ids = list(result.scalars().all())
    if not ids:
        print("📄 No queued documents to process", flush=True)
        return
    print(f"⏳ Resuming {len(ids)} queued document(s)...", flush=True)
    for document_id in ids:
        await process_uploaded_document(document_id, app)


def _run_parse_and_embed(
    pipeline: DocumentPipeline, file_path: str, fallback_title: str
):
    parsed = pipeline.parse(file_path, fallback_title)
    embeddings = pipeline.embed_chunks(parsed)
    return parsed, embeddings


@router.get("", response_model=list[LibraryDocumentResponse])
@router.get("/", response_model=list[LibraryDocumentResponse], include_in_schema=False)
async def list_documents(session: SessionDep, identity: IdentityDep):
    chunk_counts = (
        select(
            LibraryDocumentChunk.document_id,
            func.count(LibraryDocumentChunk.id).label("chunk_count"),
        )
        .group_by(LibraryDocumentChunk.document_id)
        .subquery()
    )
    result = await session.execute(
        select(LibraryDocument, func.coalesce(chunk_counts.c.chunk_count, 0))
        .outerjoin(chunk_counts, chunk_counts.c.document_id == LibraryDocument.id)
        .where(space_clause(LibraryDocument.space_id, identity.space_ids))
        .order_by(LibraryDocument.updated_at.desc())
    )
    return [to_response(document, int(count)) for document, count in result.all()]


@router.get("/{document_id}", response_model=LibraryDocumentResponse)
async def get_document(
    document_id: uuid.UUID, session: SessionDep, identity: IdentityDep
):
    document = await session.get(LibraryDocument, document_id)
    require_visible(document, identity, kind="Document")
    return to_response(document, await chunk_count_for(session, document.id))


@router.get("/{document_id}/file")
async def download_document_file(
    document_id: uuid.UUID, session: SessionDep, identity: IdentityDep
):
    document = await session.get(LibraryDocument, document_id)
    require_visible(document, identity, kind="Document")
    path = Path(document.original_file)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Original file is missing")
    return FileResponse(
        path,
        media_type=document.mime_type or "application/octet-stream",
        filename=document.original_filename,
    )


@router.patch("/{document_id}", response_model=LibraryDocumentResponse)
async def update_document(
    document_id: uuid.UUID,
    payload: LibraryDocumentUpdate,
    session: SessionDep,
    identity: IdentityDep,
):
    document = await session.get(LibraryDocument, document_id)
    require_visible(document, identity, kind="Document")
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(document, field, value)
    document.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(document)
    return to_response(document, await chunk_count_for(session, document.id))


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID, session: SessionDep, identity: IdentityDep
):
    document = await session.get(LibraryDocument, document_id)
    require_visible(document, identity, kind="Document")
    path = Path(document.original_file)
    await session.delete(document)
    await session.commit()
    if path.exists():
        path.unlink()
