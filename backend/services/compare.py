import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Paper, PaperComparison, PaperStatus
from schemas import (
    CitationSourceType,
    CompareCitation,
    CompareLinkedNote,
    ComparePaperResult,
    CompareRequest,
    CompareResponse,
    CompareSynthesis,
    normalize_dimensions,
    normalize_paper_ids,
)
from services.agent.telemetry import log_warning
from services.embeddings import EmbeddingService
from services.llm_chat import max_tokens_from_env as max_tokens_from_env
from services.retrieval import (
    LinkedNote,
    RetrievalHit,
    list_linked_notes,
    search_for_paper,
    snippet_from,
)
from services.workflow import (
    ProgressSink,
    Step,
    WorkflowJournal,
    WorkflowRun,
    fingerprint,
    require_keys,
)

MAP_PROMPT_FILE = Path(__file__).resolve().parent.parent / "compare_map_prompt.txt"
REDUCE_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent / "compare_reduce_prompt.txt"
)
MAP_PROMPT = MAP_PROMPT_FILE.read_text().strip()
REDUCE_PROMPT = REDUCE_PROMPT_FILE.read_text().strip()
PROMPT_VERSION = "compare-papers-v1"
COMPARE_MAX_TOKENS = 8192
EVIDENCE_CHARS = 800

PAPER_LABEL = "Paper"
NOTE_LABELS = {
    "voice": "Researcher's voice note (not a paper claim)",
    "handwritten": "Researcher's handwritten note (not a paper claim)",
}


class CompareError(Exception):
    """Raised when comparison generation fails."""


class CompareValidationError(Exception):
    """Raised when the comparison request is invalid."""


def dimension_query(dimensions: list[str]) -> str:
    labels = ", ".join(dim.replace("_", " ") for dim in dimensions)
    return f"Compare these papers on: {labels}"


def _location(hit: RetrievalHit) -> str:
    parts: list[str] = []
    if hit.year is not None:
        parts.append(str(hit.year))
    if hit.page is not None:
        parts.append(f"p. {hit.page}")
    if hit.section:
        parts.append(hit.section)
    if not parts:
        return ""
    return " (" + ", ".join(parts) + ")"


def _hit_label(hit: RetrievalHit) -> str:
    if hit.source_type == "paper":
        return PAPER_LABEL
    return NOTE_LABELS.get(hit.source_type, hit.source_type)


def format_paper_evidence(paper_title: str, hits: list[RetrievalHit]) -> str:
    numbered = list(enumerate(hits, start=1))
    paper_hits = [(idx, hit) for idx, hit in numbered if hit.source_type == "paper"]
    note_hits = [(idx, hit) for idx, hit in numbered if hit.source_type != "paper"]
    blocks = [f"Paper: {paper_title}"]

    if paper_hits:
        paper_blocks = [
            f"[{idx}] {PAPER_LABEL} — {hit.title}{_location(hit)}\n"
            f"{hit.snippet(max_chars=EVIDENCE_CHARS)}"
            for idx, hit in paper_hits
        ]
        blocks.append("Paper evidence:\n" + "\n\n".join(paper_blocks))
    else:
        blocks.append("Paper evidence:\nNo paper chunks were retrieved.")

    if note_hits:
        note_blocks = [
            f"[{idx}] {_hit_label(hit)} — {hit.title}{_location(hit)}\n"
            f"{hit.snippet(max_chars=EVIDENCE_CHARS)}"
            for idx, hit in note_hits
        ]
        blocks.append(
            "Researcher notes (commentary only; do not treat as paper claims):\n"
            + "\n\n".join(note_blocks)
        )
    return "\n\n".join(blocks)


def _cell_value(payload: dict, dimension: str) -> str:
    if dimension in payload:
        return str(payload[dimension] or "").strip()
    alt = dimension.replace("_", " ")
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in (dimension.lower(), alt.lower()):
        if key in lowered:
            return str(lowered[key] or "").strip()
    return ""


def cells_from_payload(payload: dict, dimensions: list[str]) -> dict[str, str]:
    return {dimension: _cell_value(payload, dimension) for dimension in dimensions}


def _as_uuid(value: object) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


class CompareService:
    def __init__(
        self,
        llm_client: OpenAI,
        llm_model: str,
        embeddings: EmbeddingService,
        max_tokens: int = COMPARE_MAX_TOKENS,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.embeddings = embeddings
        self.max_tokens = max_tokens
        self.prompt_version = PROMPT_VERSION

    def _complete(self, messages: list[dict[str, str]]) -> str:
        from services.llm_chat import (
            effort_from_env,
            message_text,
            with_chat_extras,
        )

        effort = effort_from_env(
            "COMPARE_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
        )
        response = self.llm_client.chat.completions.create(
            **with_chat_extras(
                {
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                },
                effort=effort,
            )
        )
        return message_text(response.choices[0].message)

    def _complete_json(self, messages: list[dict[str, str]], *, what: str) -> dict:
        from services.llm_chat import complete_json_object, effort_from_env

        last_error: Exception | None = None
        working = list(messages)
        effort = effort_from_env(
            "COMPARE_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
        )
        schema = {"type": "object"}
        for attempt in range(2):
            try:
                return complete_json_object(
                    self.llm_client,
                    messages=working,
                    model=self.llm_model,
                    schema=schema,
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                    effort=effort,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                working.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON matching the schema. "
                            f"Error: {exc}. Return ONLY the JSON object."
                        ),
                    }
                )
                log_warning(
                    "compare_json_invalid",
                    stage=what,
                    attempt=attempt + 1,
                    error=str(exc),
                )
        raise CompareError(f"Could not parse {what} JSON: {last_error}") from last_error

    def _map_paper(
        self,
        paper: Paper,
        dimensions: list[str],
        hits: list[RetrievalHit],
    ) -> dict[str, str]:
        user_content = (
            f"Dimensions (use these exact keys):\n{json.dumps(dimensions)}\n\n"
            f"Evidence:\n{format_paper_evidence(paper.title, hits)}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": MAP_PROMPT},
            {"role": "user", "content": user_content},
        ]
        if not any(hit.source_type == "paper" for hit in hits):
            return dict.fromkeys(dimensions, "")
        payload = self._complete_json(messages, what="map")
        return cells_from_payload(payload, dimensions)

    def _reduce(
        self,
        papers: list[Paper],
        dimensions: list[str],
        cells: dict[uuid.UUID, dict[str, str]],
    ) -> CompareSynthesis:
        rows = []
        for paper in papers:
            rows.append(
                {
                    "paper_id": str(paper.id),
                    "title": paper.title,
                    "year": paper.year,
                    "values": cells.get(paper.id, {}),
                }
            )
        user_content = (
            f"Dimensions:\n{json.dumps(dimensions)}\n\n"
            f"Per-paper cells:\n{json.dumps(rows, indent=2)}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": REDUCE_PROMPT},
            {"role": "user", "content": user_content},
        ]
        payload = self._complete_json(messages, what="reduce")
        try:
            return CompareSynthesis.model_validate(
                {
                    "agreements": str(payload.get("agreements") or "").strip(),
                    "disagreements": str(payload.get("disagreements") or "").strip(),
                    "research_gap": str(payload.get("research_gap") or "").strip(),
                }
            )
        except ValidationError as exc:
            raise CompareError(f"Invalid synthesis payload: {exc}") from exc

    def _default_run_id(self, paper_ids: list[uuid.UUID], dimensions: list[str]) -> str:
        """Same papers and dimensions means the same run, so a repeat resumes.

        Derived rather than random on purpose: a researcher who retries a
        failed comparison from the UI sends the same request, and that should
        reuse the calls the first attempt already paid for.
        """
        digest = fingerprint(
            "compare",
            sorted(str(paper_id) for paper_id in paper_ids),
            dimensions,
            self.prompt_version,
            self.llm_model,
        )
        return f"compare:{digest[:32]}"

    async def _reduce_step(
        self,
        workflow: WorkflowRun,
        papers: list[Paper],
        dimensions: list[str],
        cells: dict[uuid.UUID, dict[str, str]],
    ) -> CompareSynthesis:
        async def run() -> dict:
            synthesis = await asyncio.to_thread(self._reduce, papers, dimensions, cells)
            return synthesis.model_dump()

        step = Step(
            kind="compare_reduce",
            label="synthesis",
            run=run,
            # The synthesis depends on every cell, so a changed cell has to
            # invalidate it -- otherwise a resumed run would pair fresh cells
            # with a stale conclusion.
            inputs=(
                dimensions,
                {str(key): value for key, value in sorted(cells.items())},
                self.prompt_version,
            ),
            validate=require_keys("agreements", "disagreements", "research_gap"),
        )
        outcome = await workflow.step(step)
        if not outcome.ok or outcome.result is None:
            raise CompareError(f"Could not synthesise the comparison: {outcome.error}")
        try:
            return CompareSynthesis.model_validate(outcome.result)
        except ValidationError as exc:
            raise CompareError(f"Invalid synthesis payload: {exc}") from exc

    async def _load_papers(
        self, session: AsyncSession, paper_ids: list[uuid.UUID]
    ) -> list[Paper]:
        result = await session.execute(select(Paper).where(Paper.id.in_(paper_ids)))
        found = {paper.id: paper for paper in result.scalars().all()}
        papers: list[Paper] = []
        missing: list[str] = []
        not_ready: list[str] = []
        for paper_id in paper_ids:
            paper = found.get(paper_id)
            if paper is None:
                missing.append(str(paper_id))
                continue
            if paper.processing_status != PaperStatus.ready.value:
                not_ready.append(paper.title or str(paper_id))
                continue
            papers.append(paper)
        if missing:
            raise CompareValidationError("Paper not found: " + ", ".join(missing))
        if not_ready:
            raise CompareValidationError(
                "Only ready papers can be compared: " + ", ".join(not_ready)
            )
        return papers

    def _map_step(
        self,
        paper: Paper,
        dimensions: list[str],
        hits: list[RetrievalHit],
    ) -> Step:
        """One paper's cells, journalled under a key that survives resume.

        The key covers the paper, the dimensions, and the prompt version --
        change any of them and this becomes a different step whose old answer
        no longer applies.
        """

        async def run() -> dict:
            cells = await asyncio.to_thread(self._map_paper, paper, dimensions, hits)
            return {"cells": cells}

        return Step(
            kind="compare_map",
            label=paper.title or str(paper.id),
            run=run,
            inputs=(str(paper.id), dimensions, self.prompt_version),
            validate=require_keys("cells"),
        )

    async def compare(
        self,
        session: AsyncSession,
        payload: CompareRequest,
        *,
        run_id: str | None = None,
        on_progress: ProgressSink | None = None,
    ) -> CompareResponse:
        """Build a comparison. Pass the same `run_id` again to resume one.

        The per-paper model calls used to run one after another, so a failure
        on the last paper discarded every call before it. They now run
        concurrently and each result is journalled, which means a retry costs
        only the steps that did not finish.
        """
        try:
            paper_ids = normalize_paper_ids(payload.paper_ids)
            dimensions = normalize_dimensions(payload.dimensions)
        except ValueError as exc:
            raise CompareValidationError(str(exc)) from exc

        papers = await self._load_papers(session, paper_ids)
        query = dimension_query(dimensions)
        query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
        linked = await list_linked_notes(session, paper_ids)
        notes_by_paper: dict[uuid.UUID, list[LinkedNote]] = {}
        for note in linked:
            notes_by_paper.setdefault(note.paper_id, []).append(note)

        workflow = WorkflowRun(
            journal=WorkflowJournal(
                session, run_id or self._default_run_id(paper_ids, dimensions)
            ),
            on_progress=on_progress,
        )
        await workflow.start()

        workflow.phase("evidence")
        hits_by_paper: dict[uuid.UUID, list[RetrievalHit]] = {}
        for position, paper in enumerate(papers, start=1):
            hits_by_paper[paper.id] = await search_for_paper(
                session, query_embedding, paper.id, query_text=query
            )
            workflow.report(paper.title, position, len(papers), cached=False)

        workflow.phase("per-paper")
        outcomes = await workflow.parallel(
            [
                self._map_step(paper, dimensions, hits_by_paper[paper.id])
                for paper in papers
            ]
        )

        cells: dict[uuid.UUID, dict[str, str]] = {}
        failures: list[str] = []
        for paper, outcome in zip(papers, outcomes, strict=True):
            if outcome.ok and outcome.result is not None:
                cells[paper.id] = cells_from_payload(
                    outcome.result.get("cells") or {}, dimensions
                )
            else:
                failures.append(f"{paper.title}: {outcome.error}")
        if failures:
            # The finished steps stay journalled, so retrying this run_id only
            # recomputes these.
            raise CompareError(
                "Could not extract cells for "
                + "; ".join(failures)
                + f" (retry run_id={workflow.journal.run_id} to reuse the rest)"
            )

        citations: list[CompareCitation] = []
        index = 1
        for paper in papers:
            for hit in hits_by_paper[paper.id]:
                citations.append(
                    CompareCitation(
                        index=index,
                        source_type=CitationSourceType(hit.source_type),
                        source_id=hit.source_id,
                        chunk_id=hit.chunk_id,
                        paper_id=paper.id,
                        title=hit.title,
                        page=hit.page,
                        section=hit.section,
                        year=hit.year,
                        snippet=snippet_from(hit.text),
                        similarity=hit.similarity,
                    )
                )
                index += 1

        workflow.phase("synthesis")
        synthesis = await self._reduce_step(workflow, papers, dimensions, cells)
        await workflow.finish()

        paper_results = [
            ComparePaperResult(
                paper_id=paper.id,
                title=paper.title,
                year=paper.year,
                values=cells.get(paper.id, dict.fromkeys(dimensions, "")),
                linked_notes=[
                    CompareLinkedNote(
                        id=note.id,
                        paper_id=note.paper_id,
                        title=note.title,
                        source_type=CitationSourceType(note.source_type),
                    )
                    for note in notes_by_paper.get(paper.id, [])
                ],
            )
            for paper in papers
        ]

        created_at = datetime.now(UTC)
        comparison = PaperComparison(
            paper_ids=[str(paper_id) for paper_id in paper_ids],
            dimensions=dimensions,
            result={
                "papers": [item.model_dump(mode="json") for item in paper_results],
                "synthesis": synthesis.model_dump(),
            },
            citations=[item.model_dump(mode="json") for item in citations],
            model=self.llm_model,
            prompt_version=self.prompt_version,
            created_at=created_at,
        )
        session.add(comparison)
        await session.commit()
        await session.refresh(comparison)
        return response_from_row(comparison)


def response_from_row(row: PaperComparison) -> CompareResponse:
    result = row.result or {}
    papers_raw = result.get("papers") or []
    papers = [ComparePaperResult.model_validate(item) for item in papers_raw]
    synthesis = CompareSynthesis.model_validate(result.get("synthesis") or {})
    citations = [CompareCitation.model_validate(item) for item in (row.citations or [])]
    paper_ids = [_as_uuid(item) for item in row.paper_ids]
    return CompareResponse(
        id=row.id,
        paper_ids=paper_ids,
        dimensions=list(row.dimensions or []),
        papers=papers,
        synthesis=synthesis,
        citations=citations,
        model=row.model,
        prompt_version=row.prompt_version,
        created_at=row.created_at,
    )
