import asyncio
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
from services.embeddings import EmbeddingService
from services.retrieval import (
    DEFAULT_MIN_SIMILARITY,
    LibraryPaper,
    RetrievalHit,
    clamp_top_k,
    list_ready_papers,
    search,
)

PROMPT_FILE = Path(__file__).resolve().parent.parent / "ask_research_prompt.txt"
ASK_PROMPT = PROMPT_FILE.read_text().strip()
PROMPT_VERSION = "ask-research-v2"
ASK_MAX_TOKENS = 1500
EVIDENCE_CHARS = 800

SOURCE_LABELS = {
    "paper": "Paper",
    "voice": "Voice note",
    "handwritten": "Handwritten note",
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
        evidence_header = f"[{index}] {label} — {hit.title}{_location(hit)}"
        body = hit.snippet(max_chars=EVIDENCE_CHARS)
        blocks.append(f"{evidence_header}\n{body}")
    return f"{header}\n\n" + "\n\n".join(blocks)


class AskService:
    def __init__(
        self,
        llm_client: OpenAI,
        llm_model: str,
        embeddings: EmbeddingService,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.embeddings = embeddings
        self.min_similarity = min_similarity
        self.prompt_version = PROMPT_VERSION

    def _complete(self, messages: list[dict[str, str]]) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
            max_tokens=ASK_MAX_TOKENS,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()

    def _generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        decision: AskDecision,
        papers: list[LibraryPaper],
    ) -> str:
        max_similarity = decision.max_similarity
        similarity_line = (
            "none"
            if max_similarity is None
            else f"{max_similarity:.2f} (threshold {self.min_similarity:.2f})"
        )
        user_content = (
            f"Question:\n{question}\n\n"
            f"Evidence quality: {quality_instruction(decision.quality)}\n"
            f"Max similarity: {similarity_line}\n\n"
            f"Evidence:\n{format_evidence(hits, papers)}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": ASK_PROMPT},
            {"role": "user", "content": user_content},
        ]
        output = self._complete(messages)
        if not output:
            raise AskError("The model returned an empty answer")
        return output

    async def ask(self, session: AsyncSession, payload: AskRequest) -> AskResponse:
        top_k = clamp_top_k(payload.top_k)
        query_embedding = await asyncio.to_thread(
            self.embeddings.embed_query, payload.question
        )
        papers = await list_ready_papers(session)
        hits = await search(
            session,
            query_embedding,
            include_papers=payload.include_papers,
            include_voice_notes=payload.include_voice_notes,
            include_handwritten_notes=payload.include_handwritten_notes,
            top_k=top_k,
        )
        decision = decide_ask(
            payload.question,
            hits,
            papers,
            self.min_similarity,
        )
        if decision.skip_llm:
            answer = abstain_answer(payload.question, decision)
        else:
            answer = await asyncio.to_thread(
                self._generate,
                payload.question,
                list(decision.generation_hits),
                decision,
                papers,
            )
        citations = [
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
            for index, hit in enumerate(decision.citation_hits, start=1)
        ]
        return AskResponse(
            answer=answer,
            insufficient_evidence=decision.insufficient_evidence,
            max_similarity=decision.max_similarity,
            citations=citations,
            model=self.llm_model,
            prompt_version=self.prompt_version,
            query_kind=decision.query_kind,
            library_coverage=decision.coverage,
            suggest_external_search=decision.suggest_external_search,
            external_search_status=ExternalSearchStatus.unavailable,
        )
