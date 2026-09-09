from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    LibraryDocument,
    Note,
    NoteSourceType,
    Paper,
    PaperStatus,
    ProcessingStatus,
    ReviewStatus,
)
from services.document_pipeline import document_file_path
from services.note_pipeline import note_file_path
from services.paper_pipeline import paper_file_path

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
ALLOWED_DOCUMENT_SUFFIXES = {".pdf", ".md", ".txt", ".csv", ".docx"}


def validate_pdf_bytes(
    filename: str | None,
    content_type: str | None,
    content: bytes,
    *,
    fallback_name: str,
) -> str:
    name = filename or fallback_name
    suffix = Path(name).suffix.lower()
    ctype = (content_type or "").lower()
    if suffix != ".pdf" and "pdf" not in ctype:
        raise ValueError("Please upload a PDF file")
    if not content:
        raise ValueError("Uploaded file is empty")
    if len(content) > MAX_PDF_BYTES:
        raise ValueError("PDF is larger than 50MB")
    return name


async def create_pending_paper(
    session: AsyncSession,
    content: bytes,
    filename: str,
) -> Paper:
    paper_id = uuid.uuid4()
    dest = paper_file_path(paper_id)
    dest.write_bytes(content)
    paper = Paper(
        id=paper_id,
        title=Path(filename).stem or "Untitled paper",
        authors=[],
        tags=[],
        original_filename=filename,
        original_file=str(dest),
        processing_status=PaperStatus.pending.value,
    )
    session.add(paper)
    await session.commit()
    await session.refresh(paper)
    return paper


async def create_pending_handwritten_note(
    session: AsyncSession,
    content: bytes,
    filename: str,
) -> Note:
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
    return note


def validate_library_file(
    filename: str | None,
    content_type: str | None,
    content: bytes,
    *,
    fallback_name: str,
) -> str:
    name = filename or fallback_name
    suffix = Path(name).suffix.lower()
    ctype = (content_type or "").lower()
    if suffix not in ALLOWED_DOCUMENT_SUFFIXES and "pdf" not in ctype:
        raise ValueError(
            "Please attach a PDF, Markdown, text, CSV, or Word (.docx) file"
        )
    if ctype.startswith("audio/"):
        raise ValueError(
            "Audio belongs in Voice notes. Attach a PDF or a document instead."
        )
    if not content:
        raise ValueError("Uploaded file is empty")
    limit = MAX_PDF_BYTES if suffix == ".pdf" or "pdf" in ctype else MAX_DOCUMENT_BYTES
    if len(content) > limit:
        raise ValueError("File is larger than 50MB")
    return name


async def create_pending_document(
    session: AsyncSession,
    content: bytes,
    filename: str,
    mime_type: str | None = None,
) -> LibraryDocument:
    document_id = uuid.uuid4()
    dest = document_file_path(document_id, filename)
    dest.write_bytes(content)
    document = LibraryDocument(
        id=document_id,
        title=Path(filename).stem or "Untitled document",
        original_filename=filename,
        original_file=str(dest),
        mime_type=mime_type,
        processing_status=ProcessingStatus.pending.value,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document
