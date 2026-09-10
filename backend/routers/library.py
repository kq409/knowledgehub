from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from routers.documents import schedule_document_processing
from routers.documents import to_response as document_to_response
from routers.notes import schedule_note_processing
from routers.notes import to_response as note_to_response
from routers.papers import schedule_paper_processing
from routers.papers import to_response as paper_to_response
from schemas import LibraryUploadResponse
from services.document_classifier import classify_attachment
from services.identity import IdentityDep
from services.library_ingest import (
    create_pending_document,
    create_pending_handwritten_note,
    create_pending_paper,
    validate_library_file,
    validate_pdf_bytes,
)

router = APIRouter(prefix="/api/library", tags=["library"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.post("/upload", response_model=LibraryUploadResponse, status_code=202)
async def upload_library_pdf(
    request: Request,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
    identity: IdentityDep,
    kind: Annotated[str | None, Form()] = None,
):
    from services.demo import require_uploads

    require_uploads()
    content = await file.read()
    suggested = await asyncio.to_thread(
        classify_attachment,
        filename=file.filename or "document.pdf",
        content=content,
        content_type=file.content_type,
        prompt="",
    )
    chosen = (kind or "").strip().lower()
    resolved = chosen if chosen in {"paper", "note", "document"} else suggested

    space_id = identity.primary_space_id
    try:
        if resolved == "paper":
            filename = validate_pdf_bytes(
                file.filename,
                file.content_type,
                content,
                fallback_name="document.pdf",
            )
            paper = await create_pending_paper(
                session, content, filename, space_id=space_id
            )
            print(
                f"⏳ Queued paper for processing: {paper.title} ({paper.id})",
                flush=True,
            )
            schedule_paper_processing(request.app, paper.id)
            return LibraryUploadResponse(
                kind="paper",
                paper=paper_to_response(paper, 0),
                suggested_kind=suggested,
            )
        if resolved == "note":
            filename = validate_pdf_bytes(
                file.filename,
                file.content_type,
                content,
                fallback_name="document.pdf",
            )
            note = await create_pending_handwritten_note(
                session, content, filename, space_id=space_id
            )
            print(
                f"⏳ Queued PDF note for processing: {note.title} ({note.id})",
                flush=True,
            )
            schedule_note_processing(request.app, note.id)
            return LibraryUploadResponse(
                kind="note",
                note=note_to_response(note, 0),
                suggested_kind=suggested,
            )
        filename = validate_library_file(
            file.filename,
            file.content_type,
            content,
            fallback_name="document.pdf",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document = await create_pending_document(
        session,
        content,
        filename,
        mime_type=file.content_type,
        space_id=space_id,
    )
    print(
        f"⏳ Queued document for processing: {document.title} ({document.id})",
        flush=True,
    )
    schedule_document_processing(request.app, document.id)
    return LibraryUploadResponse(
        kind="document",
        document=document_to_response(document, 0),
        suggested_kind=suggested,
    )
