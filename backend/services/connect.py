import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from models import Note, NoteLinkSource, NotePaper, NoteSourceType, ProcessingStatus
from schemas import (
    ConnectReasonMode,
    ConnectRequest,
    ConnectResponse,
    RelatedPaper,
)
from services.embeddings import EmbeddingService
from services.extraction import parse_json_object
from services.llm_chat import (
    effort_from_env,
    max_tokens_from_env,
    message_text,
    with_chat_extras,
)
from services.note_links import (
    LINK_SOURCE_AI,
    MAX_PAPERS_PER_NOTE,
    link_note_paper,
    list_links_for_note,
    list_skipped_paper_ids,
)
from services.note_pipeline import voice_text_for_embedding
from services.retrieval import (
    DEFAULT_MIN_SIMILARITY,
    MAX_TOP_K,
    RetrievalHit,
    clamp_top_k,
    min_similarity_from_env,
    search,
)

PROMPT_FILE = Path(__file__).resolve().parent.parent / "connect_note_prompt.txt"
CONNECT_PROMPT = PROMPT_FILE.read_text().strip()
PROMPT_VERSION = "connect-note-v1"
CONNECT_MAX_TOKENS = 2048
DEFAULT_CONNECT_TOP_K = 5
MAX_CONNECT_TOP_K = 8
NOTE_CHARS = 2000
SNIPPET_CHARS = 400


class ConnectError(Exception):
    """Raised when related-paper search fails unexpectedly."""


class ConnectNotFoundError(Exception):
    """Raised when the note does not exist."""


class ConnectNotReadyError(Exception):
    """Raised when the note is not ready to search from."""


@dataclass(frozen=True)
class PaperCandidate:
    paper_id: uuid.UUID
    title: str
    year: int | None
    similarity: float
    snippet: str
    page: int | None
    section: str | None
    chunk_id: uuid.UUID


def connect_min_similarity_from_env() -> float:
    raw = os.getenv("CONNECT_MIN_SIMILARITY")
    if raw is None or not raw.strip():
        return min_similarity_from_env()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MIN_SIMILARITY
    return max(0.0, min(value, 1.0))


def clamp_connect_top_k(top_k: int | None) -> int:
    if top_k is None:
        return DEFAULT_CONNECT_TOP_K
    return max(1, min(top_k, MAX_CONNECT_TOP_K))


def query_text_for_note(note: Note) -> str:
    if note.source_type == NoteSourceType.handwritten.value:
        return (note.extracted_text or "").strip()
    return voice_text_for_embedding(
        title=note.title,
        summary=note.summary,
        observations=note.observations or [],
        hypotheses=note.hypotheses or [],
        questions=note.questions or [],
        next_steps=note.next_steps or [],
        cleaned_transcript=note.cleaned_transcript,
    ).strip()


def aggregate_paper_hits(hits: list[RetrievalHit]) -> list[PaperCandidate]:
    best: dict[uuid.UUID, RetrievalHit] = {}
    for hit in hits:
        if hit.source_type != "paper":
            continue
        current = best.get(hit.source_id)
        if current is None or hit.similarity > current.similarity:
            best[hit.source_id] = hit
    ranked = sorted(best.values(), key=lambda hit: hit.similarity, reverse=True)
    return [
        PaperCandidate(
            paper_id=hit.source_id,
            title=hit.title,
            year=hit.year,
            similarity=hit.similarity,
            snippet=hit.snippet(max_chars=SNIPPET_CHARS),
            page=hit.page,
            section=hit.section,
            chunk_id=hit.chunk_id,
        )
        for hit in ranked
    ]


def select_candidates(
    ranked: list[PaperCandidate],
    *,
    min_similarity: float,
    skipped: set[uuid.UUID],
    top_k: int,
) -> list[PaperCandidate]:
    kept = [
        item
        for item in ranked
        if item.similarity >= min_similarity and item.paper_id not in skipped
    ]
    return kept[:top_k]


def papers_to_auto_link(
    candidates: list[PaperCandidate],
    *,
    already_linked: set[uuid.UUID],
    remaining_slots: int,
) -> list[PaperCandidate]:
    if remaining_slots <= 0:
        return []
    to_add = [item for item in candidates if item.paper_id not in already_linked]
    return to_add[:remaining_slots]


def apply_reasons(
    candidates: list[PaperCandidate],
    reasons: dict[uuid.UUID, str],
    *,
    reason_mode: ConnectReasonMode,
) -> list[tuple[PaperCandidate, str]]:
    results: list[tuple[PaperCandidate, str]] = []
    for item in candidates:
        llm_reason = reasons.get(item.paper_id, "").strip()
        if reason_mode is ConnectReasonMode.llm and llm_reason:
            results.append((item, llm_reason))
        else:
            results.append((item, item.snippet))
    return results


def parse_reason_payload(payload: dict) -> dict[uuid.UUID, str]:
    rows = payload.get("reasons")
    if not isinstance(rows, list):
        return {}
    parsed: dict[uuid.UUID, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("paper_id")
        reason = str(row.get("reason") or "").strip()
        if not raw_id or not reason:
            continue
        try:
            parsed[uuid.UUID(str(raw_id))] = reason
        except ValueError:
            continue
    return parsed


def response_from_stored(note: Note) -> ConnectResponse | None:
    payload = note.related_result
    if not isinstance(payload, dict) or not payload:
        return None
    try:
        return ConnectResponse.model_validate(payload)
    except Exception:
        return None


def empty_connect_response(
    note_id: uuid.UUID,
    *,
    reason_mode: ConnectReasonMode,
    linked_paper_ids: list[uuid.UUID],
    skipped_paper_ids: list[uuid.UUID],
    model: str | None,
    generated_at: datetime | None = None,
) -> ConnectResponse:
    return ConnectResponse(
        note_id=note_id,
        reason_mode=reason_mode,
        papers=[],
        linked_paper_ids=linked_paper_ids,
        skipped_paper_ids=skipped_paper_ids,
        model=model,
        prompt_version=PROMPT_VERSION,
        generated_at=generated_at or datetime.now(UTC),
    )


class ConnectService:
    def __init__(
        self,
        llm_client: OpenAI,
        llm_model: str,
        embeddings: EmbeddingService,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        max_tokens: int = CONNECT_MAX_TOKENS,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.embeddings = embeddings
        self.min_similarity = min_similarity
        self.max_tokens = max_tokens
        self.prompt_version = PROMPT_VERSION

    def _complete(self, messages: list[dict[str, str]]) -> str:
        effort = effort_from_env(
            "CONNECT_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
        )
        max_tokens = max_tokens_from_env("CONNECT_MAX_TOKENS", self.max_tokens)
        response = self.llm_client.chat.completions.create(
            **with_chat_extras(
                {
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                effort=effort,
            )
        )
        return message_text(response.choices[0].message)

    def _llm_reasons(
        self, note_text: str, candidates: list[PaperCandidate]
    ) -> dict[uuid.UUID, str]:
        if not candidates:
            return {}
        papers = [
            {
                "paper_id": str(item.paper_id),
                "title": item.title,
                "year": item.year,
                "snippet": item.snippet,
            }
            for item in candidates
        ]
        user_content = (
            f"Note:\n{note_text[:NOTE_CHARS]}\n\n"
            f"Candidate papers:\n{json.dumps(papers, ensure_ascii=False)}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": CONNECT_PROMPT},
            {"role": "user", "content": user_content},
        ]
        last_error: Exception | None = None
        working = list(messages)
        for attempt in range(2):
            output = self._complete(working)
            try:
                payload = parse_json_object(output)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                working.append({"role": "assistant", "content": output})
                working.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON matching the schema. "
                            f"Error: {exc}. Return ONLY the JSON object."
                        ),
                    }
                )
                print(f"⚠️  Connect reasons attempt {attempt + 1} failed: {exc}")
                continue
            return parse_reason_payload(payload)
        print(f"⚠️  Connect reasons fell back to snippets: {last_error}")
        return {}

    async def _persist(
        self, session: AsyncSession, note: Note, response: ConnectResponse
    ) -> ConnectResponse:
        note.related_result = response.model_dump(mode="json")
        note.updated_at = datetime.now(UTC)
        await session.commit()
        return response

    async def connect(
        self,
        session: AsyncSession,
        note_id: uuid.UUID,
        payload: ConnectRequest | None = None,
    ) -> ConnectResponse:
        request = payload or ConnectRequest()
        note = await session.get(Note, note_id)
        if note is None:
            raise ConnectNotFoundError(f"Note not found: {note_id}")
        if note.processing_status != ProcessingStatus.ready.value:
            raise ConnectNotReadyError("Note is not ready yet")

        top_k = clamp_connect_top_k(request.top_k)
        reason_mode = request.reason_mode
        existing_links = await list_links_for_note(session, note.id)
        linked_by_id = {row.paper_id: row for row in existing_links}
        skipped = await list_skipped_paper_ids(session, note.id)
        linked_ids = [row.paper_id for row in existing_links]
        skipped_ids = list(skipped)

        query_text = query_text_for_note(note)
        if not query_text:
            return await self._persist(
                session,
                note,
                empty_connect_response(
                    note.id,
                    reason_mode=reason_mode,
                    linked_paper_ids=linked_ids,
                    skipped_paper_ids=skipped_ids,
                    model=self.llm_model,
                ),
            )

        query_embedding = await asyncio.to_thread(
            self.embeddings.embed_query, query_text
        )
        chunk_top_k = clamp_top_k(min(MAX_TOP_K, max(8, top_k * 3)))
        hits = await search(
            session,
            query_embedding,
            query_text=query_text,
            include_papers=True,
            include_voice_notes=False,
            include_handwritten_notes=False,
            include_documents=False,
            top_k=chunk_top_k,
            expand_links=False,
        )
        ranked = aggregate_paper_hits(hits)
        candidates = select_candidates(
            ranked,
            min_similarity=self.min_similarity,
            skipped=skipped,
            top_k=top_k,
        )

        reasons: dict[uuid.UUID, str] = {}
        if reason_mode is ConnectReasonMode.llm and candidates:
            try:
                reasons = await asyncio.to_thread(
                    self._llm_reasons, query_text, candidates
                )
            except Exception as exc:
                print(f"⚠️  Connect LLM reasons failed: {exc}")
                reasons = {}

        explained = apply_reasons(candidates, reasons, reason_mode=reason_mode)
        used_llm = reason_mode is ConnectReasonMode.llm and any(
            reasons.get(item.paper_id, "").strip() for item, _reason in explained
        )
        effective_mode = (
            ConnectReasonMode.llm if used_llm else ConnectReasonMode.snippet
        )

        to_link = papers_to_auto_link(
            candidates,
            already_linked=set(linked_by_id),
            remaining_slots=MAX_PAPERS_PER_NOTE - len(linked_by_id),
        )
        reason_by_id = {item.paper_id: reason for item, reason in explained}
        for item in to_link:
            await link_note_paper(
                session,
                note.id,
                item.paper_id,
                source=LINK_SOURCE_AI,
                similarity=item.similarity,
                snippet=item.snippet,
                reason=reason_by_id.get(item.paper_id, item.snippet),
                reason_mode=effective_mode.value,
                commit=False,
            )

        # Refresh AI metadata on existing AI links from this run.
        for item, reason in explained:
            row = await session.get(NotePaper, (note.id, item.paper_id))
            if row is None or row.source != LINK_SOURCE_AI:
                continue
            row.similarity = item.similarity
            row.snippet = item.snippet
            row.reason = reason
            row.reason_mode = effective_mode.value

        refreshed = await list_links_for_note(session, note.id)
        linked_by_id = {row.paper_id: row for row in refreshed}
        papers = [
            RelatedPaper(
                paper_id=item.paper_id,
                title=item.title,
                year=item.year,
                similarity=item.similarity,
                snippet=item.snippet,
                reason=reason,
                reason_mode=(
                    ConnectReasonMode(linked_by_id[item.paper_id].reason_mode)
                    if item.paper_id in linked_by_id
                    and linked_by_id[item.paper_id].reason_mode
                    else effective_mode
                ),
                linked=item.paper_id in linked_by_id,
                source=(
                    NoteLinkSource(linked_by_id[item.paper_id].source)
                    if item.paper_id in linked_by_id
                    else None
                ),
                page=item.page,
                section=item.section,
            )
            for item, reason in explained
        ]
        response = ConnectResponse(
            note_id=note.id,
            reason_mode=effective_mode if papers else reason_mode,
            papers=papers,
            linked_paper_ids=[row.paper_id for row in refreshed],
            skipped_paper_ids=sorted(skipped),
            model=self.llm_model,
            prompt_version=self.prompt_version,
            generated_at=datetime.now(UTC),
        )
        return await self._persist(session, note, response)
