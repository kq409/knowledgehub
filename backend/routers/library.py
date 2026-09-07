from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from routers.notes import schedule_note_processing
from routers.notes import to_response as note_to_response
from routers.papers import schedule_paper_processing
from routers.papers import to_response as paper_to_response
from schemas import LibraryUploadResponse
from services.document_classifier import classify_pdf
from services.library_ingest import (
    create_pending_handwritten_note,
    create_pending_paper,
    validate_pdf_bytes,
)

router = APIRouter(prefix="/api/library", tags=["library"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.post("/upload", response_model=LibraryUploadResponse, status_code=202)
async def upload_library_pdf(
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
            fallback_name="document.pdf",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    kind = await asyncio.to_thread(classify_pdf, content)
    if kind == "paper":
        paper = await create_pending_paper(session, content, filename)
        print(
            f"⏳ Queued paper for processing: {paper.title} ({paper.id})",
            flush=True,
        )
        schedule_paper_processing(request.app, paper.id)
        return LibraryUploadResponse(kind="paper", paper=paper_to_response(paper, 0))

    note = await create_pending_handwritten_note(session, content, filename)
    print(
        f"⏳ Queued PDF note for processing: {note.title} ({note.id})",
        flush=True,
    )
    schedule_note_processing(request.app, note.id)
    return LibraryUploadResponse(kind="note", note=note_to_response(note, 0))
