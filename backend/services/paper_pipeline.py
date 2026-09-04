import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Paper, PaperChunk, PaperStatus
from services.embeddings import EmbeddingService
from services.paper_parser import PaperParser, ParsedPaper

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def papers_dir() -> Path:
    configured = os.getenv("PAPERS_DIR", "data/papers")
    path = Path(configured)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def paper_file_path(paper_id: uuid.UUID) -> Path:
    return papers_dir() / f"{paper_id}.pdf"


class PaperPipeline:
    def __init__(self, parser: PaperParser, embeddings: EmbeddingService):
        self.parser = parser
        self.embeddings = embeddings

    def parse_and_embed(self, pdf_path: str, fallback_title: str) -> ParsedPaper:
        return self.parser.parse(pdf_path, fallback_title)

    def embed_chunks(self, parsed: ParsedPaper) -> list[list[float]]:
        return [self.embeddings.embed_document(chunk.text) for chunk in parsed.chunks]


async def apply_parse_result(
    session: AsyncSession,
    paper: Paper,
    parsed: ParsedPaper,
    embeddings: list[list[float]],
) -> Paper:
    await session.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper.id))
    paper.title = parsed.title or paper.title
    if parsed.authors:
        paper.authors = parsed.authors
    if parsed.year is not None:
        paper.year = parsed.year
    paper.abstract = parsed.abstract
    paper.page_count = parsed.page_count
    paper.processing_status = PaperStatus.ready.value
    paper.processing_error = None
    paper.updated_at = datetime.now(UTC)

    for index, (chunk, embedding) in enumerate(
        zip(parsed.chunks, embeddings, strict=True)
    ):
        session.add(
            PaperChunk(
                paper_id=paper.id,
                chunk_index=index,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                embedding=embedding,
                extra={},
            )
        )
    await session.commit()
    await session.refresh(paper)
    return paper


async def mark_paper_status(
    session: AsyncSession,
    paper_id: uuid.UUID,
    status: PaperStatus,
    error: str | None = None,
) -> None:
    paper = await session.get(Paper, paper_id)
    if paper is None:
        return
    paper.processing_status = status.value
    paper.processing_error = error
    paper.updated_at = datetime.now(UTC)
    await session.commit()


async def chunk_count_for(session: AsyncSession, paper_id: uuid.UUID) -> int:
    result = await session.execute(
        select(func.count(PaperChunk.id)).where(PaperChunk.paper_id == paper_id)
    )
    return int(result.scalar_one())
