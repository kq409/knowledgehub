import os
import uuid
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Note, NoteChunk, ProcessingStatus
from services.accession import apply_processing_outcome, mark_processing_fields
from services.chunking import ParsedChunk, split_with_overlap
from services.embeddings import EmbeddingService
from services.note_parser import NotePdfParser, ParsedNotePdf

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

VOICE_SECTION = "Voice Note"
DOCUMENT_SECTION = "Document"


def notes_dir() -> Path:
    configured = os.getenv("NOTES_DIR", "data/notes")
    path = Path(configured)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def note_file_path(note_id: uuid.UUID) -> Path:
    return notes_dir() / f"{note_id}.pdf"


def voice_text_for_embedding(
    *,
    title: str,
    summary: str,
    observations: list[str],
    hypotheses: list[str],
    questions: list[str],
    next_steps: list[str],
    cleaned_transcript: str,
) -> str:
    sections = [title.strip(), summary.strip()]
    labeled = [
        ("Observations", observations),
        ("Hypotheses", hypotheses),
        ("Questions", questions),
        ("Next steps", next_steps),
    ]
    for label, items in labeled:
        cleaned_items = [item.strip() for item in items if item and item.strip()]
        if cleaned_items:
            sections.append(f"{label}: " + "; ".join(cleaned_items))
    if cleaned_transcript.strip():
        sections.append(cleaned_transcript.strip())
    return "\n\n".join(part for part in sections if part)


def chunks_from_text(text: str, *, page: int | None, section: str) -> list[ParsedChunk]:
    return [
        ParsedChunk(text=part, page=page, section=section[:512])
        for part in split_with_overlap(text)
    ]


def chunks_from_voice_note(note: Note) -> list[ParsedChunk]:
    text = voice_text_for_embedding(
        title=note.title,
        summary=note.summary,
        observations=note.observations or [],
        hypotheses=note.hypotheses or [],
        questions=note.questions or [],
        next_steps=note.next_steps or [],
        cleaned_transcript=note.cleaned_transcript,
    )
    return chunks_from_text(text, page=None, section=VOICE_SECTION)


def chunks_from_extracted_text(text: str) -> list[ParsedChunk]:
    return chunks_from_text(text, page=None, section=DOCUMENT_SECTION)


class NotePipeline:
    def __init__(self, parser: NotePdfParser, embeddings: EmbeddingService):
        self.parser = parser
        self.embeddings = embeddings

    def parse_pdf(self, pdf_path: str, fallback_title: str) -> ParsedNotePdf:
        return self.parser.parse(pdf_path, fallback_title)

    def embed_chunks(self, chunks: list[ParsedChunk]) -> list[list[float]]:
        return [self.embeddings.embed_document(chunk.text) for chunk in chunks]


async def replace_note_chunks(
    session: AsyncSession,
    note: Note,
    chunks: list[ParsedChunk],
    embeddings: list[list[float]],
) -> None:
    await session.execute(delete(NoteChunk).where(NoteChunk.note_id == note.id))
    session.expire(note, ["chunks"])
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        session.add(
            NoteChunk(
                note_id=note.id,
                chunk_index=index,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                embedding=embedding,
                extra={},
            )
        )


async def apply_pdf_parse_result(
    session: AsyncSession,
    note: Note,
    parsed: ParsedNotePdf,
    embeddings: list[list[float]],
) -> Note:
    note.title = parsed.title or note.title
    note.extracted_text = parsed.extracted_text
    note.page_count = parsed.page_count
    apply_processing_outcome(note, chunk_count=len(parsed.chunks))
    await replace_note_chunks(session, note, parsed.chunks, embeddings)
    await session.commit()
    await session.refresh(note)
    return note


async def apply_text_chunks(
    session: AsyncSession,
    note: Note,
    chunks: list[ParsedChunk],
    embeddings: list[list[float]],
) -> Note:
    apply_processing_outcome(note, chunk_count=len(chunks))
    await replace_note_chunks(session, note, chunks, embeddings)
    await session.commit()
    await session.refresh(note)
    return note


async def mark_note_status(
    session: AsyncSession,
    note_id: uuid.UUID,
    status: ProcessingStatus,
    error: str | None = None,
) -> None:
    note = await session.get(Note, note_id)
    if note is None:
        return
    mark_processing_fields(note, status.value, error)
    await session.commit()


async def chunk_count_for(session: AsyncSession, note_id: uuid.UUID) -> int:
    result = await session.execute(
        select(func.count(NoteChunk.id)).where(NoteChunk.note_id == note_id)
    )
    return int(result.scalar_one())
