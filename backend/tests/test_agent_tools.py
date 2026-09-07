import os
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

import db
from models import (
    EMBEDDING_DIM,
    Note,
    NoteChunk,
    NotePaper,
    NoteSourceType,
    Paper,
    PaperChunk,
    PaperStatus,
    ProcessingStatus,
    ReviewStatus,
)
from schemas import (
    DEFAULT_COMPARE_DIMENSIONS,
    CitationSourceType,
    CompareCitation,
    ComparePaperResult,
    CompareResponse,
    CompareSynthesis,
    ExtractedNote,
)
from services.agent.tools import (
    NOTE_DISCLAIMER,
    TOOL_HANDLERS,
    TOOL_SCHEMAS,
    CitationRegistry,
    ToolContext,
    ToolError,
    compare_papers,
    connect_note,
    link_note,
    list_notes,
    list_papers,
    present_workspace,
    preview_note_extraction,
    read_note,
    read_paper,
    search_library,
    tool_catalog,
    unlink_note,
)
from services.agent.tools import (
    web_search as web_search_tool,
)
from services.compare import CompareError
from services.extraction import NoteExtractionError
from services.web_search import ExternalHit, WebSearchError

load_dotenv()

DIM = EMBEDDING_DIM


def unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


@dataclass
class Library:
    session: AsyncSession
    ctx: ToolContext
    paper: Paper
    note: Note


@pytest.fixture
async def library() -> AsyncGenerator[Library, None]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")

    db.init_db(database_url)
    try:
        if db.SessionLocal is None:
            pytest.skip("Database session factory was not created")
        async with db.SessionLocal() as probe:
            await probe.execute(text("SELECT 1 FROM paper_chunks LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL is not available")

    now = datetime.now(UTC)
    paper = Paper(
        id=uuid.uuid4(),
        title="ELLA: Efficient Lifelong Learning",
        authors=["Ruvolo", "Eaton"],
        year=2006,
        tags=[],
        original_filename="ella.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        created_at=now,
        updated_at=now,
    )
    chunks = [
        PaperChunk(
            id=uuid.uuid4(),
            paper_id=paper.id,
            chunk_index=index,
            text=body,
            page=index + 1,
            section=section,
            embedding=unit(index),
            extra={},
            created_at=now,
        )
        for index, (body, section) in enumerate(
            [
                ("ELLA transfers knowledge across tasks efficiently.", "Abstract"),
                ("The method maintains a shared basis of latent components.", "Method"),
                (
                    "Experiments cover land mine detection and facial expression.",
                    "Results",
                ),
            ]
        )
    ]
    note = Note(
        id=uuid.uuid4(),
        source_type=NoteSourceType.voice.value,
        title="Doubts about the ELLA baseline",
        summary="The baseline choice looks weak to me.",
        observations=["Only three datasets"],
        hypotheses=["A stronger baseline would close the gap"],
        questions=["Does this hold on modern benchmarks?"],
        next_steps=["Re-run with a 2024 baseline"],
        tags=["lifelong-learning"],
        raw_transcript="raw",
        cleaned_transcript="The baseline choice looks weak to me.",
        review_status=ReviewStatus.reviewed.value,
        processing_status=ProcessingStatus.ready.value,
        created_at=now,
        updated_at=now,
    )
    note_chunk = NoteChunk(
        id=uuid.uuid4(),
        note_id=note.id,
        chunk_index=0,
        text="The baseline choice looks weak to me.",
        page=None,
        section="Voice Note",
        embedding=unit(1),
        extra={},
        created_at=now,
    )

    async with db.SessionLocal() as session:
        session.add(paper)
        await session.flush()
        for chunk in chunks:
            session.add(chunk)
        session.add(note)
        await session.flush()
        session.add(
            NotePaper(
                note_id=note.id,
                paper_id=paper.id,
                created_at=now,
            )
        )
        session.add(note_chunk)
        await session.commit()

    embeddings = MagicMock()
    embeddings.embed_query.return_value = unit(0)

    try:
        async with db.SessionLocal() as session:
            yield Library(
                session=session,
                ctx=ToolContext(
                    session=session,
                    embeddings=embeddings,
                    registry=CitationRegistry(),
                ),
                paper=paper,
                note=note,
            )
    finally:
        async with db.SessionLocal() as session:
            db_note = await session.get(Note, note.id)
            if db_note is not None:
                await session.delete(db_note)
            db_paper = await session.get(Paper, paper.id)
            if db_paper is not None:
                await session.delete(db_paper)
            await session.commit()
        await db.close_db()


def test_every_schema_has_a_handler():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert names == set(TOOL_HANDLERS)


def test_catalog_describes_arguments():
    catalog = tool_catalog()
    assert "search_library(query: string" in catalog
    assert "list_papers(no arguments)" in catalog
    assert "present_workspace(module: string)" in catalog


async def test_list_papers_reports_coverage(library: Library):
    result = await list_papers(library.ctx)

    assert str(library.paper.id) in result.content
    assert "2006" in result.content
    assert "3 chunk(s)" in result.content
    assert "paper(s)" in result.summary


async def test_list_notes_marks_notes_as_the_researchers_own(library: Library):
    result = await list_notes(library.ctx, source_type="voice")

    assert str(library.note.id) in result.content
    assert "not published paper claims" in result.content
    assert "review=reviewed" in result.content
    assert "linked to ELLA" in result.content


async def test_list_notes_rejects_unknown_source_type(library: Library):
    with pytest.raises(ToolError, match="source_type must be one of"):
        await list_notes(library.ctx, source_type="sketches")


async def test_search_library_numbers_and_registers_evidence(library: Library):
    result = await search_library(library.ctx, query="lifelong learning transfer")

    assert "[1]" in result.content
    assert "similarity" in result.content
    citations = library.ctx.registry.citations()
    assert citations
    assert citations[0].index == 1
    assert citations[0].similarity is not None


async def test_search_library_labels_notes_distinctly(library: Library):
    result = await search_library(
        library.ctx, query="baseline doubts", sources=["voice_notes"]
    )

    assert NOTE_DISCLAIMER in result.content
    assert all(
        citation.source_type.value == "voice"
        for citation in library.ctx.registry.citations()
    )


async def test_search_library_requires_a_query(library: Library):
    with pytest.raises(ToolError, match="query is required"):
        await search_library(library.ctx, query="   ")


async def test_search_library_rejects_unknown_sources(library: Library):
    with pytest.raises(ToolError, match="Unknown sources"):
        await search_library(library.ctx, query="ella", sources=["arxiv"])


async def test_search_library_respects_disabled_sources(library: Library):
    library.ctx.include_papers = False
    library.ctx.include_voice_notes = False
    library.ctx.include_handwritten_notes = False

    with pytest.raises(ToolError, match="disabled every source"):
        await search_library(library.ctx, query="ella")


async def test_read_paper_returns_source_order_and_offers_the_next_slice(
    library: Library,
):
    result = await read_paper(library.ctx, paper_id=str(library.paper.id), limit=2)

    assert "chunk 0" in result.content
    assert "chunk 1" in result.content
    assert "start_index=2" in result.content
    citations = library.ctx.registry.citations()
    assert [citation.index for citation in citations] == [1, 2, 3]
    assert citations[2].source_type.value == "voice"
    assert all(citation.similarity is None for citation in citations)


async def test_read_paper_reports_the_end_of_the_paper(library: Library):
    result = await read_paper(
        library.ctx, paper_id=str(library.paper.id), start_index=2
    )

    assert "End of paper" in result.content


async def test_read_paper_rejects_a_bad_id(library: Library):
    with pytest.raises(ToolError, match="must be a UUID"):
        await read_paper(library.ctx, paper_id="the ELLA paper")


async def test_read_paper_reports_a_missing_paper(library: Library):
    with pytest.raises(ToolError, match="No paper with id"):
        await read_paper(library.ctx, paper_id=str(uuid.uuid4()))


async def test_read_note_carries_the_disclaimer_and_fields(library: Library):
    result = await read_note(library.ctx, note_id=str(library.note.id))

    assert NOTE_DISCLAIMER in result.content
    assert "Hypotheses: A stronger baseline" in result.content
    assert "Review status: reviewed" in result.content
    assert "Linked papers: ELLA" in result.content
    assert library.ctx.registry.citations()[0].similarity is None


async def test_read_note_respects_disabled_note_types(library: Library):
    library.ctx.include_voice_notes = False

    with pytest.raises(ToolError, match="disabled voice notes"):
        await read_note(library.ctx, note_id=str(library.note.id))


def compare_citation(
    paper_id: uuid.UUID, chunk_id: uuid.UUID | None = None, index: int = 1
) -> CompareCitation:
    return CompareCitation(
        index=index,
        source_type=CitationSourceType.paper,
        source_id=paper_id,
        chunk_id=chunk_id or uuid.uuid4(),
        paper_id=paper_id,
        title=f"Paper {index}",
        page=1,
        section="Abstract",
        year=2006,
        snippet="Shared latent basis across tasks.",
        similarity=0.71,
    )


def compare_response(
    paper_ids: list[uuid.UUID],
    citations: list[CompareCitation],
    dimensions: tuple[str, ...] = ("method", "dataset"),
) -> CompareResponse:
    return CompareResponse(
        id=uuid.uuid4(),
        paper_ids=paper_ids,
        dimensions=list(dimensions),
        papers=[
            ComparePaperResult(
                paper_id=paper_id,
                title=f"Paper {number}",
                year=2006,
                values={dim: f"{dim} of paper {number}" for dim in dimensions},
            )
            for number, paper_id in enumerate(paper_ids, start=1)
        ],
        synthesis=CompareSynthesis(
            agreements="Both build a shared basis.",
            disagreements="They differ on the datasets.",
            research_gap="Neither tests modern benchmarks.",
        ),
        citations=citations,
        model="test-model",
        prompt_version="compare-papers-v1",
        created_at=datetime.now(UTC),
    )


def compare_service(response: CompareResponse) -> MagicMock:
    service = MagicMock()
    service.compare = AsyncMock(return_value=response)
    return service


async def test_compare_papers_hands_the_model_registry_numbers(library: Library):
    ids = [uuid.uuid4(), uuid.uuid4()]
    citations = [compare_citation(ids[0], index=1), compare_citation(ids[1], index=2)]
    library.ctx.compare = compare_service(compare_response(ids, citations))

    result = await compare_papers(library.ctx, paper_ids=[str(item) for item in ids])

    assert "method of paper 1" in result.content
    assert "Both build a shared basis." in result.content
    assert "Evidence you may cite:" in result.content
    assert "Paper 1: cite [1]" in result.content
    assert "Paper 2: cite [2]" in result.content
    assert [citation.index for citation in library.ctx.registry.citations()] == [1, 2]


async def test_compare_papers_artifact_matches_the_turns_citation_numbers(
    library: Library,
):
    """Compare numbers its own evidence from 1; the chat must show one scheme."""
    await search_library(library.ctx, query="lifelong learning transfer")
    already_read = library.ctx.registry.citations()
    assert already_read

    ids = [uuid.uuid4(), uuid.uuid4()]
    fresh = compare_citation(ids[1], index=2)
    citations = [
        compare_citation(ids[0], chunk_id=already_read[0].chunk_id, index=1),
        fresh,
    ]
    library.ctx.compare = compare_service(compare_response(ids, citations))

    result = await compare_papers(library.ctx, paper_ids=[str(item) for item in ids])

    assert result.artifact is not None
    assert result.artifact["kind"] == "comparison"
    artifact_numbers = {
        citation["chunk_id"]: citation["index"]
        for citation in result.artifact["data"]["citations"]
    }
    registry_numbers = {
        str(citation.chunk_id): citation.index
        for citation in library.ctx.registry.citations()
    }
    assert artifact_numbers == {
        chunk_id: registry_numbers[chunk_id] for chunk_id in artifact_numbers
    }
    # The chunk the agent had already read keeps the number it was given then.
    assert artifact_numbers[str(already_read[0].chunk_id)] == already_read[0].index


async def test_compare_papers_defaults_to_the_standard_dimensions(library: Library):
    ids = [uuid.uuid4(), uuid.uuid4()]
    library.ctx.compare = compare_service(compare_response(ids, []))

    await compare_papers(library.ctx, paper_ids=[str(item) for item in ids])

    request = library.ctx.compare.compare.await_args.args[1]
    assert request.dimensions == list(DEFAULT_COMPARE_DIMENSIONS)


async def test_compare_papers_spends_its_budget_once(library: Library):
    ids = [uuid.uuid4(), uuid.uuid4()]
    library.ctx.compare = compare_service(compare_response(ids, []))
    paper_ids = [str(item) for item in ids]

    await compare_papers(library.ctx, paper_ids=paper_ids)

    with pytest.raises(ToolError, match="already run one comparison"):
        await compare_papers(library.ctx, paper_ids=paper_ids)
    assert library.ctx.compare.compare.await_count == 1


async def test_compare_papers_rejects_too_few_papers(library: Library):
    library.ctx.compare = compare_service(compare_response([], []))

    with pytest.raises(ToolError, match="at least 2"):
        await compare_papers(library.ctx, paper_ids=[str(library.paper.id)])


async def test_compare_papers_rejects_ids_that_are_not_uuids(library: Library):
    library.ctx.compare = compare_service(compare_response([], []))

    with pytest.raises(ToolError, match="must be a UUID"):
        await compare_papers(library.ctx, paper_ids=["ELLA", "the other one"])


async def test_compare_papers_reports_a_failed_comparison(library: Library):
    ids = [uuid.uuid4(), uuid.uuid4()]
    service = MagicMock()
    service.compare = AsyncMock(side_effect=CompareError("no JSON in output"))
    library.ctx.compare = service

    with pytest.raises(ToolError, match="could not be built"):
        await compare_papers(library.ctx, paper_ids=[str(item) for item in ids])


async def test_compare_papers_needs_the_service(library: Library):
    with pytest.raises(ToolError, match="not available"):
        await compare_papers(
            library.ctx, paper_ids=[str(uuid.uuid4()), str(uuid.uuid4())]
        )


async def test_preview_note_extraction_uses_the_apps_own_fields(library: Library):
    extraction = MagicMock()
    extraction.extract.return_value = ExtractedNote(
        title="Baseline doubts",
        summary="The baseline looks weak.",
        observations=["Only three datasets"],
        hypotheses=["A stronger baseline closes the gap"],
        questions=["Does this hold today?"],
        next_steps=["Re-run with a 2024 baseline"],
        tags=["lifelong-learning"],
    )
    library.ctx.extraction = extraction

    result = await preview_note_extraction(library.ctx, text="rambling thoughts")

    assert "Title: Baseline doubts" in result.content
    assert "Only three datasets" in result.content
    assert "Tags: lifelong-learning" in result.content
    assert "Nothing was saved" in result.content
    assert result.artifact is None


async def test_preview_note_extraction_requires_text(library: Library):
    library.ctx.extraction = MagicMock()

    with pytest.raises(ToolError, match="text is required"):
        await preview_note_extraction(library.ctx, text="   ")


async def test_preview_note_extraction_reports_a_failure(library: Library):
    extraction = MagicMock()
    extraction.extract.side_effect = NoteExtractionError("model returned prose")
    library.ctx.extraction = extraction

    with pytest.raises(ToolError, match="could not be structured"):
        await preview_note_extraction(library.ctx, text="rambling thoughts")


async def test_preview_note_extraction_needs_the_service(library: Library):
    with pytest.raises(ToolError, match="not available"):
        await preview_note_extraction(library.ctx, text="rambling thoughts")


async def test_connect_note_needs_the_service(library: Library):
    with pytest.raises(ToolError, match="not available"):
        await connect_note(library.ctx, note_id=str(library.note.id))


async def test_citation_numbers_are_stable_across_tool_calls(library: Library):
    first = await search_library(library.ctx, query="lifelong learning")
    count_after_first = len(library.ctx.registry.citations())

    second = await search_library(library.ctx, query="lifelong learning")

    assert count_after_first == len(library.ctx.registry.citations())
    assert first.content.count("[1]") == second.content.count("[1]")


async def test_link_note_and_unlink_note(library: Library):
    other = Paper(
        title="Other paper",
        original_filename="other.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    library.session.add(other)
    await library.session.commit()
    try:
        linked = await link_note(
            library.ctx, note_id=str(library.note.id), paper_id=str(other.id)
        )
        assert "Linked note" in linked.content
        listed = await list_notes(library.ctx, source_type="voice")
        assert "Other paper" in listed.content

        again = await link_note(
            library.ctx, note_id=str(library.note.id), paper_id=str(other.id)
        )
        assert "already linked" in again.content

        removed = await unlink_note(
            library.ctx, note_id=str(library.note.id), paper_id=str(other.id)
        )
        assert "Unlinked note" in removed.content
        listed = await list_notes(library.ctx, source_type="voice")
        assert "Other paper" not in listed.content
    finally:
        stored = await library.session.get(Paper, other.id)
        if stored is not None:
            await library.session.delete(stored)
            await library.session.commit()


async def test_subagent_schema_excludes_link_tools():
    from services.agent.tools import SUBAGENT_TOOL_NAMES, schemas_for_tools

    names = {
        schema["function"]["name"] for schema in schemas_for_tools(SUBAGENT_TOOL_NAMES)
    }
    assert "link_note" not in names
    assert "unlink_note" not in names
    assert "read_paper" in names
    assert "present_workspace" not in names
    assert "web_search" not in names


async def test_present_workspace_opens_voice_notes():
    ctx = ToolContext(session=MagicMock(), embeddings=MagicMock())
    result = await present_workspace(ctx, module="voice_notes")

    assert result.artifact is not None
    assert result.artifact["kind"] == "workspace"
    assert result.artifact["tool"] == "present_workspace"
    assert result.artifact["data"]["module"] == "voice_notes"
    assert "voice notes" in result.summary.lower()


async def test_present_workspace_rejects_unknown_module():
    ctx = ToolContext(session=MagicMock(), embeddings=MagicMock())
    with pytest.raises(ToolError, match="voice_notes"):
        await present_workspace(ctx, module="compare")


class _ReadyWeb:
    def available(self) -> bool:
        return True

    def search(self, query: str) -> list[ExternalHit]:
        return [
            ExternalHit(
                url="https://arxiv.org/abs/2401.00001",
                title="Survey of continual learning",
                snippet="Recent work surveys SOTA methods.",
            )
        ]


class _DisabledWeb:
    def available(self) -> bool:
        return False

    def search(self, query: str) -> list[ExternalHit]:
        raise WebSearchError("disabled")


class _FailWeb:
    def available(self) -> bool:
        return True

    def search(self, query: str) -> list[ExternalHit]:
        raise WebSearchError("provider down")


async def test_web_search_registers_web_citations():
    ctx = ToolContext(
        session=MagicMock(), embeddings=MagicMock(), web_search=_ReadyWeb()
    )
    result = await web_search_tool(ctx, query="continual learning site:arxiv.org")
    assert "[1]" in result.content
    assert "not from the library" in result.content
    citations = ctx.registry.citations()
    assert len(citations) == 1
    assert citations[0].source_type is CitationSourceType.web
    assert citations[0].url == "https://arxiv.org/abs/2401.00001"
    assert citations[0].title == "Survey of continual learning"


async def test_web_search_not_configured_returns_message():
    ctx = ToolContext(
        session=MagicMock(), embeddings=MagicMock(), web_search=_DisabledWeb()
    )
    result = await web_search_tool(ctx, query="SOTA continual learning")
    assert "WEB_SEARCH_API_KEY" in result.content
    assert ctx.registry.citations() == []


async def test_web_search_provider_error_fails_open():
    ctx = ToolContext(
        session=MagicMock(), embeddings=MagicMock(), web_search=_FailWeb()
    )
    result = await web_search_tool(ctx, query="SOTA continual learning")
    assert "failed" in result.content.lower()
    assert ctx.registry.citations() == []


async def test_web_search_requires_query():
    ctx = ToolContext(
        session=MagicMock(), embeddings=MagicMock(), web_search=_ReadyWeb()
    )
    with pytest.raises(ToolError, match="query"):
        await web_search_tool(ctx, query="  ")
