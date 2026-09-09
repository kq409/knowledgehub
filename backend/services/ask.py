import asyncio
import time
import uuid
from pathlib import Path

from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from schemas import (
    AskCitation,
    AskRequest,
    AskResponse,
    CitationSourceType,
    ExternalSearchStatus,
)
from services.ask_policy import (
    AskDecision,
    abstain_answer,
    decide_ask,
    inventory_line,
    quality_instruction,
)
from services.ask_trace import log_ask_trace
from services.embeddings import EmbeddingService
from services.llm_chat import (
    effort_from_env,
    max_tokens_from_env,
)
from services.retrieval import (
    DEFAULT_MIN_SIMILARITY,
    LibraryPaper,
    RetrievalHit,
    clamp_top_k,
    list_ready_papers,
    search,
)
from services.web_search import (
    ExternalHit,
    WebSearcher,
    WebSearchError,
    WebSearchService,
    web_citation_id,
)

PROMPT_FILE = Path(__file__).resolve().parent.parent / "ask_research_prompt.txt"
ASK_PROMPT = PROMPT_FILE.read_text().strip()
PROMPT_VERSION = "ask-research-v3"
ASK_MAX_TOKENS_DEFAULT = 4096
EVIDENCE_CHARS = 800

SOURCE_LABELS = {
    "paper": "Paper",
    "voice": "Voice note",
    "handwritten": "Handwritten note",
    "document": "Document",
    "web": "Web",
}


class AskError(Exception):
    """Raised when the LLM does not return an answer."""


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


def format_evidence(
    hits: list[RetrievalHit], papers: list[LibraryPaper] | None = None
) -> str:
    header = inventory_line(papers or [], hits)
    if not hits:
        return f"{header}\nNo evidence was retrieved from the personal library."

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        label = SOURCE_LABELS.get(hit.source_type, hit.source_type)
        extras: list[str] = []
        if hit.linked_titles:
            extras.append("linked to " + ", ".join(hit.linked_titles))
        if hit.via_link:
            extras.append("via link")
        extra = f" ({'; '.join(extras)})" if extras else ""
        evidence_header = f"[{index}] {label} — {hit.title}{_location(hit)}{extra}"
        body = hit.snippet(max_chars=EVIDENCE_CHARS)
        blocks.append(f"{evidence_header}\n{body}")
    return f"{header}\n\n" + "\n\n".join(blocks)


def format_web_evidence(hits: list[ExternalHit], start_index: int) -> str:
    if not hits:
        return "No web evidence was retrieved."
    blocks: list[str] = []
    for offset, hit in enumerate(hits):
        index = start_index + offset
        body = hit.snippet or hit.title
        blocks.append(f"[{index}] Web — {hit.title} ({hit.url})\n{body}")
    return "Web evidence (not from the library):\n\n" + "\n\n".join(blocks)


class AskService:
    def __init__(
        self,
        llm_client: OpenAI,
        llm_model: str,
        embeddings: EmbeddingService,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        web_search: WebSearcher | None = None,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.embeddings = embeddings
        self.min_similarity = min_similarity
        self.prompt_version = PROMPT_VERSION
        self.web_search = web_search if web_search is not None else WebSearchService()
        self.last_prompt_tokens: int | None = None
        self.last_completion_tokens: int | None = None

    def library_search_status(self) -> ExternalSearchStatus:
        if self.web_search.available():
            return ExternalSearchStatus.ready
        return ExternalSearchStatus.unavailable

    def _complete(self, messages: list[dict[str, str]]) -> str:
        from services.llm_chat import complete_chat

        effort = effort_from_env(
            "ASK_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
        )
        max_tokens = max_tokens_from_env("ASK_MAX_TOKENS", ASK_MAX_TOKENS_DEFAULT)
        result = complete_chat(
            self.llm_client,
            {
                "model": self.llm_model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": False,
            },
            effort=effort,
        )
        self.last_prompt_tokens = result.prompt_tokens
        self.last_completion_tokens = result.completion_tokens
        return result.text

    def _generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        decision: AskDecision,
        papers: list[LibraryPaper],
        web_hits: list[ExternalHit] | None = None,
    ) -> str:
        max_similarity = decision.max_similarity
        similarity_line = (
            "none"
            if max_similarity is None
            else f"{max_similarity:.2f} (threshold {self.min_similarity:.2f})"
        )
        library_block = format_evidence(hits, papers)
        web = web_hits or []
        web_block = ""
        if web:
            web_block = "\n\n" + format_web_evidence(web, start_index=len(hits) + 1)
        user_content = (
            f"Question:\n{question}\n\n"
            f"Evidence quality: {quality_instruction(decision.quality)}\n"
            f"Max similarity: {similarity_line}\n\n"
            f"Evidence:\n{library_block}{web_block}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": ASK_PROMPT},
            {"role": "user", "content": user_content},
        ]
        output = self._complete(messages)
        if not output:
            raise AskError("The model returned an empty answer")
        return output

    def _library_citations(self, hits: tuple[RetrievalHit, ...]) -> list[AskCitation]:
        return [
            AskCitation(
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
            )
            for index, hit in enumerate(hits, start=1)
        ]

    def _web_citations(
        self, hits: list[ExternalHit], start_index: int
    ) -> list[AskCitation]:
        citations: list[AskCitation] = []
        for offset, hit in enumerate(hits):
            ident = web_citation_id(hit.url)
            citations.append(
                AskCitation(
                    index=start_index + offset,
                    source_type=CitationSourceType.web,
                    source_id=ident,
                    chunk_id=ident,
                    title=hit.title,
                    snippet=hit.snippet or hit.title,
                    similarity=0.0,
                    url=hit.url,
                )
            )
        return citations

    async def ask(self, session: AsyncSession, payload: AskRequest) -> AskResponse:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        top_k = clamp_top_k(payload.top_k)
        query_embedding = await asyncio.to_thread(
            self.embeddings.embed_query, payload.question
        )
        papers = await list_ready_papers(session, paper_ids=payload.paper_ids)
        hits = await search(
            session,
            query_embedding,
            query_text=payload.question,
            include_papers=payload.include_papers,
            include_voice_notes=payload.include_voice_notes,
            include_handwritten_notes=payload.include_handwritten_notes,
            top_k=top_k,
            paper_ids=payload.paper_ids,
        )
        decision = decide_ask(
            payload.question,
            hits,
            papers,
            self.min_similarity,
        )
        status = self.library_search_status()
        web_hits: list[ExternalHit] = []
        should_search_web = (
            payload.external_search
            and decision.suggest_external_search
            and self.web_search.available()
        )
        if should_search_web:
            try:
                web_hits = await asyncio.to_thread(
                    self.web_search.search, payload.question
                )
                status = ExternalSearchStatus.ran
            except WebSearchError as exc:
                print(f"⚠️  Web search failed: {exc}", flush=True)
                status = ExternalSearchStatus.failed

        skip_llm = decision.skip_llm and not web_hits
        if skip_llm:
            answer = abstain_answer(payload.question, decision)
        else:
            answer = await asyncio.to_thread(
                self._generate,
                payload.question,
                list(decision.generation_hits or decision.citation_hits),
                decision,
                papers,
                web_hits,
            )

        citations = self._library_citations(decision.citation_hits)
        citations.extend(self._web_citations(web_hits, start_index=len(citations) + 1))
        latency_ms = (time.perf_counter() - started) * 1000
        result = AskResponse(
            answer=answer,
            insufficient_evidence=decision.insufficient_evidence and not web_hits,
            max_similarity=decision.max_similarity,
            citations=citations,
            model=self.llm_model,
            prompt_version=self.prompt_version,
            query_kind=decision.query_kind,
            library_coverage=decision.coverage,
            suggest_external_search=decision.suggest_external_search,
            external_search_status=status,
            request_id=request_id,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=self.last_prompt_tokens,
            completion_tokens=self.last_completion_tokens,
        )
        log_ask_trace(
            question=payload.question,
            query_kind=decision.query_kind.value,
            hits=[
                {
                    "chunk_id": str(hit.chunk_id),
                    "title": hit.title,
                    "similarity": hit.similarity,
                    "rank_score": hit.rank_score,
                }
                for hit in hits[:16]
            ],
            decision=decision.quality.value,
            external_search=payload.external_search,
            external_status=status.value,
            web_urls=[hit.url for hit in web_hits],
            citations=[
                {
                    "index": item.index,
                    "source_type": item.source_type.value,
                    "title": item.title,
                    "url": item.url,
                }
                for item in citations
            ],
            answer=answer,
            latency_ms=latency_ms,
            request_id=request_id,
            prompt_tokens=self.last_prompt_tokens,
            completion_tokens=self.last_completion_tokens,
        )
        return result
