"""The tool pool the research agent acts through.

Most tools are read-only. `link_note` / `unlink_note` and memory writes persist
library metadata the researcher can undo in the UI. The loop never inspects
tool names: it looks handlers up in TOOL_HANDLERS and hands whatever comes back
to the model.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from models import Note, NoteSourceType, Paper
from schemas import (
    DEFAULT_COMPARE_DIMENSIONS,
    ArtifactKind,
    ChatCitation,
    CitationSourceType,
    CompareCitation,
    CompareRequest,
    CompareResponse,
    ConnectReasonMode,
    ConnectRequest,
    normalize_dimensions,
    normalize_paper_ids,
)
from services.compare import CompareError, CompareService, CompareValidationError
from services.connect import (
    ConnectError,
    ConnectNotFoundError,
    ConnectNotReadyError,
    ConnectService,
)
from services.embeddings import EmbeddingService
from services.extraction import ExtractionService, NoteExtractionError
from services.note_links import (
    NoteLinkError,
    link_note_paper,
    linked_titles_for_notes,
    notes_linked_to_paper,
    unlink_note_paper,
)
from services.retrieval import (
    DocumentChunk,
    LibraryNoteRow,
    LibraryPaperRow,
    RetrievalHit,
    clamp_top_k,
    list_library_notes,
    list_library_papers,
    read_paper_chunks,
    search,
    snippet_from,
)

SEARCH_EVIDENCE_CHARS = 800
READ_EVIDENCE_CHARS = 1500
NOTE_FIELD_ITEMS = 8
COMPARE_CELL_CHARS = 400
EXTRACT_PREVIEW_CHARS = 40000

# A comparison costs one model call per paper plus one to synthesise. One per
# turn is plenty for a conversation; deeper questions want read_paper instead.
COMPARE_CALLS_PER_TURN = 1

SOURCE_LABELS = {
    "paper": "Paper",
    "voice": "Voice note",
    "handwritten": "Handwritten note",
}
NOTE_DISCLAIMER = "Researcher's own note, not a paper claim"

PAPER_SOURCE = "papers"
VOICE_SOURCE = "voice_notes"
HANDWRITTEN_SOURCE = "handwritten_notes"
ALL_SOURCES = (PAPER_SOURCE, VOICE_SOURCE, HANDWRITTEN_SOURCE)


class ToolError(Exception):
    """Raised when a tool cannot run. The loop reports it back to the model."""


@dataclass(frozen=True)
class ToolResult:
    """Three audiences, three fields — plus optional side channels.

    `content` goes to the model and is truncated to fit the context budget.
    `summary` is the one line the UI shows in the tool trace. `artifact` is
    structured output the UI renders itself; it never enters the conversation,
    so it is neither truncated nor paid for in tokens. `todo_items` is a full
    list snapshot the loop turns into a `todo` SSE event for the panel.
    """

    content: str
    summary: str
    artifact: dict | None = None
    todo_items: list[dict] | None = None


class CitationRegistry:
    """Assigns a stable [n] to every chunk the agent reads.

    The same chunk keeps its number for the whole turn, so the model can cite
    something it saw three tool calls ago, and it can only cite what it read.
    """

    def __init__(self) -> None:
        self._indices: dict[uuid.UUID, int] = {}
        self._citations: list[ChatCitation] = []

    def _add(self, chunk_id: uuid.UUID, build: Callable[[int], ChatCitation]) -> int:
        existing = self._indices.get(chunk_id)
        if existing is not None:
            return existing
        index = len(self._citations) + 1
        self._indices[chunk_id] = index
        self._citations.append(build(index))
        return index

    def register_hit(self, hit: RetrievalHit) -> int:
        return self._add(
            hit.chunk_id,
            lambda index: ChatCitation(
                index=index,
                source_type=CitationSourceType(hit.source_type),
                source_id=hit.source_id,
                chunk_id=hit.chunk_id,
                title=hit.title,
                page=hit.page,
                section=hit.section,
                year=hit.year,
                snippet=hit.snippet(),
                similarity=hit.similarity,
            ),
        )

    def register_paper_chunk(self, paper: Paper, chunk: DocumentChunk) -> int:
        return self._add(
            chunk.chunk_id,
            lambda index: ChatCitation(
                index=index,
                source_type=CitationSourceType.paper,
                source_id=paper.id,
                chunk_id=chunk.chunk_id,
                title=paper.title,
                page=chunk.page,
                section=chunk.section,
                year=paper.year,
                snippet=snippet_from(chunk.text),
                similarity=None,
            ),
        )

    def register_note(self, note: Note, snippet: str) -> int:
        return self._add(
            note.id,
            lambda index: ChatCitation(
                index=index,
                source_type=CitationSourceType(note.source_type),
                source_id=note.id,
                chunk_id=note.id,
                title=note.title,
                page=None,
                section=None,
                year=None,
                snippet=snippet_from(snippet),
                similarity=None,
            ),
        )

    def register_compare_citation(self, citation: CompareCitation) -> int:
        """Adopt a chunk the comparison workflow retrieved on its own.

        Compare numbers its evidence from 1 for its own saved record. Feeding
        those numbers straight to the model would put two numbering schemes in
        one conversation, so the chunk is re-registered here and the caller
        renumbers. A chunk the agent already searched up keeps its first number.
        """
        return self._add(
            citation.chunk_id,
            lambda index: ChatCitation(
                index=index,
                source_type=citation.source_type,
                source_id=citation.source_id,
                chunk_id=citation.chunk_id,
                title=citation.title,
                page=citation.page,
                section=citation.section,
                year=citation.year,
                snippet=citation.snippet,
                similarity=citation.similarity,
            ),
        )

    def merge_citation(self, citation: ChatCitation) -> int:
        """Adopt a citation another agent already read (e.g. a subagent).

        Keeps the first number if this turn already saw the same chunk_id.
        """
        return self._add(
            citation.chunk_id,
            lambda index: ChatCitation(
                index=index,
                source_type=citation.source_type,
                source_id=citation.source_id,
                chunk_id=citation.chunk_id,
                title=citation.title,
                page=citation.page,
                section=citation.section,
                year=citation.year,
                snippet=citation.snippet,
                similarity=citation.similarity,
            ),
        )

    def merge_registry(self, other: CitationRegistry) -> dict[int, int]:
        """Copy every citation from `other`. Returns child_index → parent_index."""
        mapping: dict[int, int] = {}
        for citation in other.citations():
            mapping[citation.index] = self.merge_citation(citation)
        return mapping

    def citations(self) -> list[ChatCitation]:
        return list(self._citations)

    def indices(self) -> set[int]:
        """The [n] numbers the agent is entitled to cite."""
        return {citation.index for citation in self._citations}


@dataclass
class ToolContext:
    """Everything a handler needs, plus the source toggles from the request."""

    session: AsyncSession
    embeddings: EmbeddingService
    registry: CitationRegistry = field(default_factory=CitationRegistry)
    include_papers: bool = True
    include_voice_notes: bool = True
    include_handwritten_notes: bool = True
    top_k: int | None = None
    compare: CompareService | None = None
    extraction: ExtractionService | None = None
    connect: ConnectService | None = None
    compare_budget: int = COMPARE_CALLS_PER_TURN
    todo_store: object | None = None
    depth: int = 0
    subagent_budget: int = 1

    def allowed_sources(self) -> list[str]:
        allowed: list[str] = []
        if self.include_papers:
            allowed.append(PAPER_SOURCE)
        if self.include_voice_notes:
            allowed.append(VOICE_SOURCE)
        if self.include_handwritten_notes:
            allowed.append(HANDWRITTEN_SOURCE)
        return allowed


def _as_uuid(value: object, *, field_name: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value).strip())
    except (AttributeError, ValueError) as exc:
        raise ToolError(
            f"{field_name} must be a UUID from list_papers or list_notes, got {value!r}"
        ) from exc


def _as_int(
    value: object, *, field_name: str, default: int | None = None
) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{field_name} must be a whole number, got {value!r}") from exc


def _location(page: int | None, section: str | None, year: int | None) -> str:
    parts: list[str] = []
    if year is not None:
        parts.append(str(year))
    if page is not None:
        parts.append(f"p. {page}")
    if section:
        parts.append(section)
    if not parts:
        return ""
    return " (" + ", ".join(parts) + ")"


def _hit_label(hit: RetrievalHit) -> str:
    label = SOURCE_LABELS.get(hit.source_type, hit.source_type)
    if hit.source_type == "paper":
        return label
    return f"{label} — {NOTE_DISCLAIMER}"


def _requested_sources(value: object, allowed: list[str]) -> list[str]:
    if value is None:
        return allowed
    if isinstance(value, str):
        requested = [value]
    elif isinstance(value, list):
        requested = [str(item) for item in value]
    else:
        raise ToolError(f"sources must be a list, got {value!r}")

    unknown = [item for item in requested if item not in ALL_SOURCES]
    if unknown:
        raise ToolError(
            f"Unknown sources {unknown}. Valid values: {list(ALL_SOURCES)}."
        )
    return [item for item in requested if item in allowed]


async def search_library(ctx: ToolContext, **kwargs: object) -> ToolResult:
    query = str(kwargs.get("query") or "").strip()
    if not query:
        raise ToolError("query is required and cannot be empty")

    allowed = ctx.allowed_sources()
    if not allowed:
        raise ToolError("The researcher disabled every source for this conversation")

    sources = _requested_sources(kwargs.get("sources"), allowed)
    if not sources:
        raise ToolError(
            f"None of the requested sources are enabled. Enabled: {allowed}."
        )

    requested_top_k = _as_int(kwargs.get("top_k"), field_name="top_k")
    top_k = clamp_top_k(requested_top_k if requested_top_k is not None else ctx.top_k)

    query_embedding = await asyncio.to_thread(ctx.embeddings.embed_query, query)
    hits = await search(
        ctx.session,
        query_embedding,
        query_text=query,
        include_papers=PAPER_SOURCE in sources,
        include_voice_notes=VOICE_SOURCE in sources,
        include_handwritten_notes=HANDWRITTEN_SOURCE in sources,
        top_k=top_k,
    )
    if not hits:
        return ToolResult(
            content=(
                f'No chunks matched "{query}" in the enabled sources ({", ".join(sources)}). '
                "Try different wording, or use list_papers to see what the library holds."
            ),
            summary=f'Searched "{query}" — no matches',
        )

    blocks: list[str] = []
    for hit in hits:
        index = ctx.registry.register_hit(hit)
        extras: list[str] = [f"similarity {hit.similarity:.2f}"]
        if hit.linked_titles:
            extras.append("linked to " + ", ".join(hit.linked_titles))
        if hit.via_link:
            extras.append("via link")
        header = (
            f"[{index}] {_hit_label(hit)} — {hit.title}"
            f"{_location(hit.page, hit.section, hit.year)} "
            f"({'; '.join(extras)})"
        )
        blocks.append(f"{header}\n{hit.snippet(max_chars=SEARCH_EVIDENCE_CHARS)}")

    body = "\n\n".join(blocks)
    return ToolResult(
        content=f'{len(hits)} result(s) for "{query}":\n\n{body}',
        summary=f'Searched "{query}" — {len(hits)} result(s)',
    )


def _paper_line(row: LibraryPaperRow) -> str:
    year = row.year if row.year is not None else "year unknown"
    authors = ", ".join(row.authors[:3]) if row.authors else "authors unknown"
    return (
        f"- id={row.id} | {row.title} | {year} | {authors} "
        f"| status={row.processing_status} | {row.chunk_count} chunk(s)"
    )


async def list_papers(ctx: ToolContext, **_: object) -> ToolResult:
    rows = await list_library_papers(ctx.session)
    if not rows:
        return ToolResult(
            content="The paper library is empty. No paper can support any claim.",
            summary="Listed papers — library is empty",
        )

    years = sorted({row.year for row in rows if row.year is not None})
    year_part = (
        f"{years[0]}-{years[-1]}"
        if len(years) > 1
        else (str(years[0]) if years else "unknown")
    )
    ready = sum(1 for row in rows if row.processing_status == "ready")
    header = (
        f"Library: {len(rows)} paper(s), {ready} ready to search. "
        f"Publication years: {year_part}."
    )
    lines = "\n".join(_paper_line(row) for row in rows)
    return ToolResult(
        content=f"{header}\n\n{lines}",
        summary=f"Listed {len(rows)} paper(s) in the library",
    )


def _note_line(row: LibraryNoteRow) -> str:
    label = SOURCE_LABELS.get(row.source_type, row.source_type)
    linked = " | linked to " + ", ".join(row.linked_titles) if row.linked_titles else ""
    return (
        f"- id={row.id} | {row.title} | {label} | review={row.review_status} "
        f"| status={row.processing_status} | {row.chunk_count} chunk(s){linked}"
    )


async def list_notes(ctx: ToolContext, **kwargs: object) -> ToolResult:
    raw = kwargs.get("source_type")
    requested = str(raw).strip().lower() if raw is not None else "all"
    valid = {"voice", "handwritten", "all"}
    if requested not in valid:
        raise ToolError(f"source_type must be one of {sorted(valid)}, got {raw!r}")

    enabled: list[str] = []
    if ctx.include_voice_notes:
        enabled.append(NoteSourceType.voice.value)
    if ctx.include_handwritten_notes:
        enabled.append(NoteSourceType.handwritten.value)
    if requested != "all":
        enabled = [item for item in enabled if item == requested]
    if not enabled:
        raise ToolError(
            "The researcher disabled those note types for this conversation"
        )

    rows = await list_library_notes(ctx.session, source_types=enabled)
    if not rows:
        return ToolResult(
            content="No notes of that type exist.",
            summary="Listed notes — none found",
        )

    header = (
        f"{len(rows)} note(s). These are the researcher's own notes, "
        "not published paper claims."
    )
    lines = "\n".join(_note_line(row) for row in rows)
    return ToolResult(
        content=f"{header}\n\n{lines}",
        summary=f"Listed {len(rows)} note(s)",
    )


async def read_paper(ctx: ToolContext, **kwargs: object) -> ToolResult:
    if not ctx.include_papers:
        raise ToolError("The researcher disabled papers for this conversation")

    paper_id = _as_uuid(kwargs.get("paper_id"), field_name="paper_id")
    start_index = _as_int(
        kwargs.get("start_index"), field_name="start_index", default=0
    )
    limit = _as_int(kwargs.get("limit"), field_name="limit")

    paper = await ctx.session.get(Paper, paper_id)
    if paper is None:
        raise ToolError(
            f"No paper with id {paper_id}. Use list_papers to get valid ids."
        )

    page = await read_paper_chunks(
        ctx.session,
        paper_id,
        start_index=start_index or 0,
        limit=limit,
    )
    if page.total == 0:
        return ToolResult(
            content=(
                f'"{paper.title}" has no parsed text '
                f"(processing status: {paper.processing_status})."
            ),
            summary=f'Read "{paper.title}" — no parsed text',
        )
    if not page.chunks:
        return ToolResult(
            content=(
                f'"{paper.title}" has {page.total} chunk(s), so start_index '
                f"{page.start_index} is past the end."
            ),
            summary=f'Read "{paper.title}" — start_index out of range',
        )

    blocks: list[str] = []
    for chunk in page.chunks:
        index = ctx.registry.register_paper_chunk(paper, chunk)
        header = (
            f"[{index}] Paper — {paper.title}"
            f"{_location(chunk.page, chunk.section, paper.year)} "
            f"(chunk {chunk.chunk_index})"
        )
        blocks.append(
            f"{header}\n{snippet_from(chunk.text, max_chars=READ_EVIDENCE_CHARS)}"
        )

    last = page.chunks[-1].chunk_index
    more = (
        f"\n\nChunks {page.chunks[0].chunk_index}-{last} of {page.total}. "
        f"Call read_paper again with start_index={last + 1} to continue."
        if last + 1 < page.total
        else f"\n\nChunks {page.chunks[0].chunk_index}-{last} of {page.total}. End of paper."
    )
    note_block = ""
    if (start_index or 0) == 0:
        linked_notes = await notes_linked_to_paper(ctx.session, paper_id)
        shown: list[str] = []
        for note in linked_notes:
            if (
                note.source_type == NoteSourceType.voice.value
                and not ctx.include_voice_notes
            ) or (
                note.source_type == NoteSourceType.handwritten.value
                and not ctx.include_handwritten_notes
            ):
                continue
            snippet = (
                note.summary.strip() or note.cleaned_transcript.strip() or note.title
            )
            index = ctx.registry.register_note(note, snippet)
            label = SOURCE_LABELS.get(note.source_type, note.source_type)
            shown.append(
                f"[{index}] {label} — {note.title} — {NOTE_DISCLAIMER}\n"
                f"{snippet_from(snippet, max_chars=SEARCH_EVIDENCE_CHARS)}"
            )
        if shown:
            note_block = (
                "\n\nLinked notes (researcher commentary, not paper claims):\n\n"
                + "\n\n".join(shown)
            )
        else:
            note_block = "\n\nNo notes are linked to this paper."
    return ToolResult(
        content="\n\n".join(blocks) + more + note_block,
        summary=f'Read "{paper.title}" chunks {page.chunks[0].chunk_index}-{last}',
    )


def _note_field(label: str, items: list[str] | None) -> str | None:
    cleaned = [item.strip() for item in (items or []) if item and item.strip()]
    if not cleaned:
        return None
    return f"{label}: " + "; ".join(cleaned[:NOTE_FIELD_ITEMS])


async def read_note(ctx: ToolContext, **kwargs: object) -> ToolResult:
    note_id = _as_uuid(kwargs.get("note_id"), field_name="note_id")
    note = await ctx.session.get(Note, note_id)
    if note is None:
        raise ToolError(f"No note with id {note_id}. Use list_notes to get valid ids.")

    if note.source_type == NoteSourceType.voice.value and not ctx.include_voice_notes:
        raise ToolError("The researcher disabled voice notes for this conversation")
    if (
        note.source_type == NoteSourceType.handwritten.value
        and not ctx.include_handwritten_notes
    ):
        raise ToolError(
            "The researcher disabled handwritten notes for this conversation"
        )

    body = note.summary.strip() or note.cleaned_transcript.strip()
    index = ctx.registry.register_note(note, body or note.title)

    label = SOURCE_LABELS.get(note.source_type, note.source_type)
    lines = [
        f"[{index}] {label} — {note.title} — {NOTE_DISCLAIMER}",
        f"Review status: {note.review_status}",
    ]
    if note.summary.strip():
        lines.append(f"Summary: {note.summary.strip()}")
    for label_text, items in (
        ("Observations", note.observations),
        ("Hypotheses", note.hypotheses),
        ("Questions", note.questions),
        ("Next steps", note.next_steps),
    ):
        line = _note_field(label_text, items)
        if line:
            lines.append(line)
    if note.tags:
        lines.append("Tags: " + ", ".join(note.tags))
    titles = await linked_titles_for_notes(ctx.session, [note.id])
    linked = titles.get(note.id, ())
    if linked:
        lines.append("Linked papers: " + ", ".join(linked))
    if note.cleaned_transcript.strip():
        lines.append(
            "Transcript: "
            + snippet_from(note.cleaned_transcript, max_chars=READ_EVIDENCE_CHARS)
        )

    return ToolResult(
        content="\n".join(lines),
        summary=f'Read note "{note.title}"',
    )


def _as_uuid_list(value: object, *, field_name: str) -> list[uuid.UUID]:
    if isinstance(value, str):
        # Some models send one long string instead of a list.
        raw = [item for item in value.replace(",", " ").split() if item]
    elif isinstance(value, list):
        raw = value
    else:
        raise ToolError(f"{field_name} must be a list of paper ids, got {value!r}")
    return [_as_uuid(item, field_name=field_name) for item in raw]


def _compare_dimensions(value: object) -> list[str]:
    if value is None:
        return list(DEFAULT_COMPARE_DIMENSIONS)
    if isinstance(value, str):
        requested = [value]
    elif isinstance(value, list):
        requested = [str(item) for item in value]
    else:
        raise ToolError(f"dimensions must be a list of labels, got {value!r}")
    try:
        return normalize_dimensions(requested)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def _renumber(response: CompareResponse, numbers: dict[uuid.UUID, int]) -> dict:
    """The artifact the UI renders, carrying registry numbers instead of its own.

    Cells are plain prose with no [n] markers in them, so only the citation
    list needs rewriting. The row saved to the database keeps Compare's own
    numbering, which the Compare tab reads on its own terms.
    """
    renumbered = [
        citation.model_copy(update={"index": numbers[citation.chunk_id]})
        for citation in response.citations
    ]
    payload = response.model_copy(update={"citations": renumbered})
    return {
        "kind": ArtifactKind.comparison.value,
        "tool": "compare_papers",
        "data": payload.model_dump(mode="json"),
    }


def _compare_evidence_lines(
    response: CompareResponse, numbers: dict[uuid.UUID, int]
) -> list[str]:
    by_paper: dict[uuid.UUID, list[int]] = {}
    for citation in response.citations:
        by_paper.setdefault(citation.paper_id, []).append(numbers[citation.chunk_id])

    lines: list[str] = []
    for paper in response.papers:
        indices = sorted(set(by_paper.get(paper.paper_id, [])))
        cite = ", ".join(f"[{index}]" for index in indices) if indices else "none"
        lines.append(f"- {paper.title}: cite {cite}")
    return lines


async def compare_papers(ctx: ToolContext, **kwargs: object) -> ToolResult:
    if ctx.compare is None:
        raise ToolError("Comparison is not available in this conversation")
    if not ctx.include_papers:
        raise ToolError("The researcher disabled papers for this conversation")
    if ctx.compare_budget <= 0:
        raise ToolError(
            "You have already run one comparison this turn. Use read_paper to "
            "go deeper on a single paper instead of comparing again."
        )

    paper_ids = _as_uuid_list(kwargs.get("paper_ids"), field_name="paper_ids")
    dimensions = _compare_dimensions(kwargs.get("dimensions"))
    try:
        normalize_paper_ids(paper_ids)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    ctx.compare_budget -= 1
    request = CompareRequest(paper_ids=paper_ids, dimensions=dimensions)
    try:
        response = await ctx.compare.compare(ctx.session, request)
    except CompareValidationError as exc:
        raise ToolError(str(exc)) from exc
    except CompareError as exc:
        raise ToolError(f"The comparison could not be built: {exc}") from exc

    numbers = {
        citation.chunk_id: ctx.registry.register_compare_citation(citation)
        for citation in response.citations
    }

    rows: list[str] = []
    for paper in response.papers:
        year = f" ({paper.year})" if paper.year is not None else ""
        cells = [
            f"  {dimension}: {(paper.values.get(dimension) or '').strip()[:COMPARE_CELL_CHARS] or '—'}"
            for dimension in response.dimensions
        ]
        rows.append(f"{paper.title}{year}\n" + "\n".join(cells))

    body = "\n\n".join(rows)
    evidence = "\n".join(_compare_evidence_lines(response, numbers))
    titles = ", ".join(paper.title for paper in response.papers)
    content = (
        f"Compared {len(response.papers)} paper(s) on "
        f"{', '.join(response.dimensions)}.\n\n"
        f"{body}\n\n"
        f"Agreements: {response.synthesis.agreements or '—'}\n"
        f"Disagreements: {response.synthesis.disagreements or '—'}\n"
        f"Research gap: {response.synthesis.research_gap or '—'}\n\n"
        f"Evidence you may cite:\n{evidence}\n\n"
        "The Compare workspace now shows this table. Answer their question from "
        "it rather than repeating every cell."
    )
    return ToolResult(
        content=content,
        summary=f"Compared {len(response.papers)} paper(s): {titles}",
        artifact=_renumber(response, numbers),
    )


WORKSPACE_MODULES = frozenset({"voice_notes"})


async def present_workspace(_ctx: ToolContext, **kwargs: object) -> ToolResult:
    """Open an interactive workspace the researcher operates themselves."""
    module = str(kwargs.get("module") or "").strip()
    if module not in WORKSPACE_MODULES:
        raise ToolError(
            "module must be voice_notes. Use compare_papers when you already "
            "know which papers to compare."
        )
    return ToolResult(
        content=(
            "The voice notes workspace is now open. The researcher will record "
            "or type there. Do not try to capture audio yourself. Confirm "
            "briefly that the recorder is ready."
        ),
        summary="Opened voice notes",
        artifact={
            "kind": "workspace",
            "tool": "present_workspace",
            "data": {"module": module},
        },
    )


def _preview_field(label: str, items: list[str] | None) -> str | None:
    cleaned = [item.strip() for item in (items or []) if item and item.strip()]
    if not cleaned:
        return None
    return f"{label}:\n" + "\n".join(f"  - {item}" for item in cleaned)


async def preview_note_extraction(ctx: ToolContext, **kwargs: object) -> ToolResult:
    if ctx.extraction is None:
        raise ToolError("Note extraction is not available in this conversation")

    text = str(kwargs.get("text") or "").strip()
    if not text:
        raise ToolError("text is required and cannot be empty")

    try:
        note = await asyncio.to_thread(
            ctx.extraction.extract, text[:EXTRACT_PREVIEW_CHARS]
        )
    except NoteExtractionError as exc:
        raise ToolError(f"The text could not be structured as a note: {exc}") from exc

    lines = [f"Title: {note.title}", f"Summary: {note.summary}"]
    for label, items in (
        ("Observations", note.observations),
        ("Hypotheses", note.hypotheses),
        ("Questions", note.questions),
        ("Next steps", note.next_steps),
    ):
        block = _preview_field(label, items)
        if block:
            lines.append(block)
    if note.tags:
        lines.append("Tags: " + ", ".join(note.tags))
    lines.append(
        "This is a preview only. Nothing was saved. The researcher saves notes "
        "from the Notes tab."
    )
    return ToolResult(
        content="\n".join(lines),
        summary=f'Structured a note preview: "{note.title}"',
    )


async def todo_write(ctx: ToolContext, **kwargs: object) -> ToolResult:
    from services.agent.todo import TodoStore, TodoStoreError

    store = ctx.todo_store
    if not isinstance(store, TodoStore):
        raise ToolError("todo_write is not available in this context")

    try:
        store.replace(kwargs.get("items"))
    except TodoStoreError as exc:
        raise ToolError(str(exc)) from exc

    snapshot = store.snapshot()
    return ToolResult(
        content=store.format_for_model(),
        summary=f"Updated todo list ({len(snapshot)} item(s))",
        todo_items=snapshot,
    )


async def spawn_subagent(ctx: ToolContext, **kwargs: object) -> ToolResult:
    """Safety net: the loop intercepts this tool and streams nested events.

    If a call somehow reaches the generic runner, refuse clearly rather than
    silently doing nothing.
    """
    raise ToolError(
        "spawn_subagent must be handled by the agent loop; "
        "the intercept path did not run"
    )


async def list_skills_tool(ctx: ToolContext, **_: object) -> ToolResult:
    from services.agent.skills import list_skill_summaries

    skills = list_skill_summaries()
    if not skills:
        return ToolResult(
            content="No skills are installed.",
            summary="Listed skills — none found",
        )
    lines = ["Available skills (load one with load_skill):"]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description}")
    return ToolResult(
        content="\n".join(lines),
        summary=f"Listed {len(skills)} skill(s)",
    )


async def load_skill_tool(ctx: ToolContext, **kwargs: object) -> ToolResult:
    from services.agent.skills import SkillError, get_skill

    name = str(kwargs.get("name") or "").strip()
    if not name:
        raise ToolError("name is required")
    try:
        skill = get_skill(name)
    except SkillError as exc:
        raise ToolError(str(exc)) from exc
    return ToolResult(
        content=skill.full_text(),
        summary=f'Loaded skill "{skill.name}"',
        artifact={
            "kind": ArtifactKind.skill.value,
            "tool": "load_skill",
            "data": {"name": skill.name, "description": skill.description},
        },
    )


async def memory_search(ctx: ToolContext, **kwargs: object) -> ToolResult:
    from services.agent.memory import MemoryError, memory_as_dict, search_memories

    query = kwargs.get("query")
    category = kwargs.get("category")
    try:
        rows = await search_memories(
            ctx.session,
            query=str(query).strip() if query is not None else None,
            category=str(category).strip() if category is not None else None,
        )
    except MemoryError as exc:
        raise ToolError(str(exc)) from exc

    if not rows:
        return ToolResult(
            content="No memories matched. The researcher has not stored one yet, or the query missed.",
            summary="Searched memories — no matches",
        )
    blocks = []
    for row in rows:
        data = memory_as_dict(row)
        blocks.append(f"- key={data['key']} [{data['category']}]\n  {data['content']}")
    return ToolResult(
        content=f"{len(rows)} memor(ies):\n" + "\n".join(blocks),
        summary=f"Searched memories — {len(rows)} hit(s)",
    )


async def memory_write(ctx: ToolContext, **kwargs: object) -> ToolResult:
    from services.agent.memory import MemoryError, memory_as_dict, upsert_memory

    try:
        row = await upsert_memory(
            ctx.session,
            key=kwargs.get("key"),
            content=kwargs.get("content"),
            category=kwargs.get("category"),
            source_turn=kwargs.get("source_turn"),
        )
    except MemoryError as exc:
        raise ToolError(str(exc)) from exc

    data = memory_as_dict(row)
    return ToolResult(
        content=(
            f"Remembered key={data['key']} [{data['category']}]:\n{data['content']}"
        ),
        summary=f'Remembered "{data["key"]}"',
        artifact={
            "kind": ArtifactKind.memory.value,
            "tool": "memory_write",
            "data": {
                "action": "remembered",
                "key": data["key"],
                "content": data["content"],
                "category": data["category"],
            },
        },
    )


async def memory_delete(ctx: ToolContext, **kwargs: object) -> ToolResult:
    from services.agent.memory import MemoryError, delete_memory

    key = kwargs.get("key")
    try:
        deleted = await delete_memory(ctx.session, key=key)
    except MemoryError as exc:
        raise ToolError(str(exc)) from exc
    key_str = str(key or "").strip()
    if not deleted:
        return ToolResult(
            content=f"No memory with key={key_str!r} to delete.",
            summary=f'No memory "{key_str}"',
        )
    return ToolResult(
        content=f"Forgot memory key={key_str}.",
        summary=f'Forgot "{key_str}"',
        artifact={
            "kind": ArtifactKind.memory.value,
            "tool": "memory_delete",
            "data": {"action": "forgot", "key": key_str},
        },
    )


async def link_note(ctx: ToolContext, **kwargs: object) -> ToolResult:
    note_id = _as_uuid(kwargs.get("note_id"), field_name="note_id")
    paper_id = _as_uuid(kwargs.get("paper_id"), field_name="paper_id")
    try:
        action = await link_note_paper(ctx.session, note_id, paper_id)
    except NoteLinkError as exc:
        raise ToolError(str(exc)) from exc

    note = await ctx.session.get(Note, note_id)
    paper = await ctx.session.get(Paper, paper_id)
    note_title = note.title if note is not None else str(note_id)
    paper_title = paper.title if paper is not None else str(paper_id)
    if action == "exists":
        return ToolResult(
            content=f'Note "{note_title}" is already linked to "{paper_title}".',
            summary=f'Already linked "{note_title}" to "{paper_title}"',
        )
    return ToolResult(
        content=(
            f'Linked note "{note_title}" to paper "{paper_title}". '
            "The note remains researcher commentary, not a paper claim."
        ),
        summary=f'Linked "{note_title}" to "{paper_title}"',
    )


async def unlink_note(ctx: ToolContext, **kwargs: object) -> ToolResult:
    note_id = _as_uuid(kwargs.get("note_id"), field_name="note_id")
    paper_id = _as_uuid(kwargs.get("paper_id"), field_name="paper_id")
    note = await ctx.session.get(Note, note_id)
    paper = await ctx.session.get(Paper, paper_id)
    if note is None:
        raise ToolError(f"No note with id {note_id}. Use list_notes to get valid ids.")
    if paper is None:
        raise ToolError(
            f"No paper with id {paper_id}. Use list_papers to get valid ids."
        )
    removed = await unlink_note_paper(ctx.session, note_id, paper_id)
    if not removed:
        return ToolResult(
            content=f'Note "{note.title}" was not linked to "{paper.title}".',
            summary=f'No link between "{note.title}" and "{paper.title}"',
        )
    return ToolResult(
        content=f'Unlinked note "{note.title}" from paper "{paper.title}".',
        summary=f'Unlinked "{note.title}" from "{paper.title}"',
    )


async def connect_note(ctx: ToolContext, **kwargs: object) -> ToolResult:
    if ctx.connect is None:
        raise ToolError("Related-paper search is not available.")
    note_id = _as_uuid(kwargs.get("note_id"), field_name="note_id")
    mode_raw = str(kwargs.get("reason_mode") or "llm").strip().lower()
    if mode_raw not in {ConnectReasonMode.llm.value, ConnectReasonMode.snippet.value}:
        raise ToolError("reason_mode must be llm or snippet.")
    try:
        response = await ctx.connect.connect(
            ctx.session,
            note_id,
            ConnectRequest(reason_mode=ConnectReasonMode(mode_raw)),
        )
    except ConnectNotFoundError as exc:
        raise ToolError(str(exc)) from exc
    except ConnectNotReadyError as exc:
        raise ToolError(str(exc)) from exc
    except ConnectError as exc:
        raise ToolError("Related papers could not be found.") from exc

    if not response.papers:
        return ToolResult(
            content=(
                "No papers in the library were similar enough to this note. "
                "Existing links were left unchanged."
            ),
            summary="No related papers",
        )
    lines = []
    for paper in response.papers:
        status = "linked" if paper.linked else "not linked"
        lines.append(
            f"- {paper.title} (similarity {paper.similarity:.2f}, {status}): "
            f"{paper.reason}"
        )
    return ToolResult(
        content=(
            f"Found {len(response.papers)} related paper(s). "
            "AI links are suggestions the researcher can undo.\n" + "\n".join(lines)
        ),
        summary=f"Related papers: {len(response.papers)}",
    )


ToolHandler = Callable[..., Awaitable[ToolResult]]

TOOL_HANDLERS: dict[str, ToolHandler] = {
    "search_library": search_library,
    "list_papers": list_papers,
    "list_notes": list_notes,
    "read_paper": read_paper,
    "read_note": read_note,
    "compare_papers": compare_papers,
    "present_workspace": present_workspace,
    "preview_note_extraction": preview_note_extraction,
    "todo_write": todo_write,
    "spawn_subagent": spawn_subagent,
    "list_skills": list_skills_tool,
    "load_skill": load_skill_tool,
    "memory_search": memory_search,
    "memory_write": memory_write,
    "memory_delete": memory_delete,
    "link_note": link_note,
    "unlink_note": unlink_note,
    "connect_note": connect_note,
}

# Nested research: read library + skills + memory_search. No writes, spawn, or compare.
SUBAGENT_TOOL_NAMES = frozenset(
    {
        "search_library",
        "list_papers",
        "list_notes",
        "read_paper",
        "read_note",
        "list_skills",
        "load_skill",
        "memory_search",
    }
)

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "description": (
                "Semantic search over the researcher's papers and notes. "
                "Returns numbered chunks you may cite as [n]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, in natural language.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(ALL_SOURCES)},
                        "description": "Which sources to search. Defaults to all enabled ones.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many chunks to return (1-16).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_papers",
            "description": (
                "List every paper in the library with id, year, and chunk count. "
                "Use this first to judge what the library can actually support."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": (
                "List the researcher's own notes with id, title, and review status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["voice", "handwritten", "all"],
                        "description": "Which note type to list. Defaults to all.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": (
                "Read a paper's text in source order, starting at a chunk index. "
                "Use it when search snippets are too shallow to answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Paper id from list_papers or a search result.",
                    },
                    "start_index": {
                        "type": "integer",
                        "description": "First chunk index to read. Defaults to 0.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many chunks to read (1-6).",
                    },
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": (
                "Read one of the researcher's notes in full. Notes are personal "
                "commentary, never evidence of what a paper claims."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Note id from list_notes or a search result.",
                    }
                },
                "required": ["note_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_papers",
            "description": (
                "Build a side-by-side comparison table of 2-4 papers, with "
                "agreements, disagreements, and the gap between them. Costs one "
                "model call per paper, so you may use it once per turn. The "
                "Compare workspace opens with the table; you get numbered "
                "evidence to cite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2 to 4 paper ids from list_papers.",
                    },
                    "dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "What to compare on, e.g. method, dataset. Defaults to "
                            f"{', '.join(DEFAULT_COMPARE_DIMENSIONS)}."
                        ),
                    },
                },
                "required": ["paper_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "present_workspace",
            "description": (
                "Open an interactive workspace the researcher operates. "
                "Use voice_notes when they want to capture a spoken or typed "
                "research note in the recording UI; they will record after it "
                "opens. Do not wait for a particular phrasing. Prefer "
                "compare_papers when you already know which papers to compare."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "enum": ["voice_notes"],
                        "description": "Which workspace to present.",
                    }
                },
                "required": ["module"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_note_extraction",
            "description": (
                "Turn loose text into the researcher's note structure: title, "
                "summary, observations, hypotheses, questions, next steps, tags. "
                "Nothing is saved. Use it when asked to organise text into a "
                "note, so the fields match the ones the app already uses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to structure.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "Replace the current research todo list with a full updated "
                "list. Use it for multi-step work (survey several papers, then "
                "synthesise). At most one item may be in_progress. Statuses: "
                "pending, in_progress, completed, cancelled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "The complete todo list after this update.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Stable id; reuse across updates.",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "What to do.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                        "cancelled",
                                    ],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_subagent",
            "description": (
                "Run a nested research pass with its own context to dig into "
                "one paper or one sub-question. You receive a short summary "
                "and renumbered citations; the researcher sees the nested "
                "tool trail. Use once per turn for deep work, not for every "
                "search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "What the nested pass should find out.",
                    },
                    "paper_id": {
                        "type": "string",
                        "description": (
                            "Optional paper id to focus on. The nested pass "
                            "still uses tools to read; nothing is pre-loaded."
                        ),
                    },
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": (
                "List on-demand skills (citation rules, compare defaults, "
                "note schema, voice cleanup). Load one with load_skill when "
                "you need the full instructions."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "Load the full text of one skill into this turn. Use when you "
                "need detailed citation, compare, note, or cleanup guidance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name from list_skills.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Search cross-session memories (preferences, hypotheses, "
                "focus areas). Call this when past preferences might matter. "
                "Empty query returns the most recent memories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring to match in key or content.",
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "preference",
                            "hypothesis",
                            "focus",
                            "workflow",
                            "other",
                        ],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Upsert a durable memory the researcher cares about across "
                "conversations (preferred compare dimensions, research focus, "
                "accepted hypotheses). Use a short stable key."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Stable key, e.g. preferred_compare_dimensions.",
                    },
                    "content": {
                        "type": "string",
                        "description": "What to remember.",
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "preference",
                            "hypothesis",
                            "focus",
                            "workflow",
                            "other",
                        ],
                    },
                    "source_turn": {
                        "type": "string",
                        "description": "Optional short note about why this was saved.",
                    },
                },
                "required": ["key", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete one memory by key when it is wrong or obsolete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key to forget.",
                    }
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "link_note",
            "description": (
                "Link one of the researcher's notes to a paper in the library. "
                "Use after search_library or list_papers when the researcher "
                "says the note is about that paper. Notes stay commentary, "
                "never paper claims. Do not guess if more than one paper fits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Note id from list_notes or a search result.",
                    },
                    "paper_id": {
                        "type": "string",
                        "description": "Paper id from list_papers or a search result.",
                    },
                },
                "required": ["note_id", "paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unlink_note",
            "description": (
                "Remove the link between one note and one paper. The note and "
                "paper themselves are not deleted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Note id from list_notes.",
                    },
                    "paper_id": {
                        "type": "string",
                        "description": "Paper id to detach.",
                    },
                },
                "required": ["note_id", "paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "connect_note",
            "description": (
                "Find papers in the local library related to one note, explain "
                "why, and auto-link them as AI suggestions the researcher can "
                "undo. Use when the researcher asks which papers a note relates to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Note id from list_notes or a search result.",
                    },
                    "reason_mode": {
                        "type": "string",
                        "description": "llm (default) or snippet.",
                    },
                },
                "required": ["note_id"],
            },
        },
    },
]


def schemas_for_tools(names: frozenset[str] | set[str]) -> list[dict]:
    """Filter TOOL_SCHEMAS down to the named tools (used by subagents)."""
    return [schema for schema in TOOL_SCHEMAS if schema["function"]["name"] in names]


def tool_catalog(schemas: list[dict] | None = None) -> str:
    """Plain-text tool list for models without native tool calling."""
    lines: list[str] = []
    for schema in schemas if schemas is not None else TOOL_SCHEMAS:
        function = schema["function"]
        properties = function["parameters"].get("properties", {})
        required = set(function["parameters"].get("required", []))
        if properties:
            args = ", ".join(
                f"{name}{'' if name in required else '?'}: {spec.get('type', 'string')}"
                for name, spec in properties.items()
            )
        else:
            args = "no arguments"
        lines.append(f"- {function['name']}({args})\n  {function['description']}")
    return "\n".join(lines)
