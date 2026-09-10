from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    DEFAULT_SPACE_ID,
    AccessionStatus,
    LibraryDocument,
    Note,
    NoteSourceType,
    Paper,
    PaperStatus,
    ProcessingStatus,
    ReviewStatus,
)
from services.storage import ImmutableObjectError, default_storage, sha256_hex

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


def _store_original(key: str, content: bytes) -> str:
    return str(default_storage().put(key, content))


async def create_pending_paper(
    session: AsyncSession,
    content: bytes,
    filename: str,
    *,
    space_id: uuid.UUID | None = None,
) -> Paper:
    paper_id = uuid.uuid4()
    dest = _store_original(f"papers/{paper_id}/r1.pdf", content)
    paper = Paper(
        id=paper_id,
        space_id=space_id or DEFAULT_SPACE_ID,
        title=Path(filename).stem or "Untitled paper",
        authors=[],
        tags=[],
        original_filename=filename,
        original_file=dest,
        processing_status=PaperStatus.pending.value,
        revision=1,
        sha256=sha256_hex(content),
        accession_status=AccessionStatus.received.value,
    )
    session.add(paper)
    await session.commit()
    await session.refresh(paper)
    return paper


async def create_pending_handwritten_note(
    session: AsyncSession,
    content: bytes,
    filename: str,
    *,
    space_id: uuid.UUID | None = None,
) -> Note:
    note_id = uuid.uuid4()
    dest = _store_original(f"notes/{note_id}/r1.pdf", content)
    note = Note(
        id=note_id,
        space_id=space_id or DEFAULT_SPACE_ID,
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
        original_file=dest,
        review_status=ReviewStatus.accepted.value,
        processing_status=ProcessingStatus.pending.value,
        revision=1,
        sha256=sha256_hex(content),
        accession_status=AccessionStatus.received.value,
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
    *,
    space_id: uuid.UUID | None = None,
) -> LibraryDocument:
    document_id = uuid.uuid4()
    suffix = Path(filename).suffix.lower() or ".bin"
    dest = _store_original(f"documents/{document_id}/r1{suffix}", content)
    document = LibraryDocument(
        id=document_id,
        space_id=space_id or DEFAULT_SPACE_ID,
        title=Path(filename).stem or "Untitled document",
        original_filename=filename,
        original_file=dest,
        mime_type=mime_type,
        processing_status=ProcessingStatus.pending.value,
        revision=1,
        sha256=sha256_hex(content),
        accession_status=AccessionStatus.received.value,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


def _kind_for(entity: Paper | Note | LibraryDocument) -> tuple[str, str, str]:
    if isinstance(entity, Paper):
        return "papers", ".pdf", PaperStatus.pending.value
    if isinstance(entity, Note):
        return "notes", ".pdf", ProcessingStatus.pending.value
    suffix = Path(entity.original_filename or "").suffix.lower() or ".bin"
    return "documents", suffix, ProcessingStatus.pending.value


async def append_revision(
    session: AsyncSession,
    entity: Paper | Note | LibraryDocument,
    content: bytes,
    filename: str,
) -> int:
    """Store a new original. Never overwrites the previous bytes."""
    kind, suffix, pending = _kind_for(entity)
    if isinstance(entity, LibraryDocument):
        suffix = Path(filename).suffix.lower() or suffix
    next_rev = int(entity.revision or 1) + 1
    key = f"{kind}/{entity.id}/r{next_rev}{suffix}"
    try:
        dest = _store_original(key, content)
    except ImmutableObjectError:
        raise
    entity.revision = next_rev
    entity.sha256 = sha256_hex(content)
    entity.original_file = dest
    entity.original_filename = filename
    entity.accession_status = AccessionStatus.received.value
    entity.processing_status = pending
    entity.processing_error = None
    entity.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(entity)
    return next_rev


def verify_checksum(entity: Paper | Note | LibraryDocument) -> bool:
    stored = getattr(entity, "sha256", None)
    path = getattr(entity, "original_file", None)
    if not stored or not path:
        return False
    data = Path(path).read_bytes()
    return sha256_hex(data) == stored
