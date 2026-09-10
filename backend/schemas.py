import uuid
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from services.note_links import MAX_PAPERS_PER_NOTE, dedupe_paper_ids


class ReviewStatus(str, Enum):
    generated = "generated"
    draft = "draft"
    reviewed = "reviewed"
    accepted = "accepted"


class ExtractedNote(BaseModel):
    title: str
    summary: str
    observations: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    raw_text: str
    cleaned_text: str


class ExtractResponse(ExtractedNote):
    model: str
    prompt_version: str


class NoteSourceType(str, Enum):
    voice = "voice"
    handwritten = "handwritten"


class NoteStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class AccessionStatus(str, Enum):
    received = "received"
    accessioned = "accessioned"
    rejected = "rejected"


def _validate_note_paper_ids(value: list[uuid.UUID]) -> list[uuid.UUID]:
    unique = dedupe_paper_ids(value)
    if len(unique) > MAX_PAPERS_PER_NOTE:
        raise ValueError(f"A note can link to at most {MAX_PAPERS_PER_NOTE} papers")
    return unique


class NoteCreate(ExtractedNote):
    raw_transcript: str
    cleaned_transcript: str
    review_status: ReviewStatus = ReviewStatus.generated
    model: str | None = None
    prompt_version: str | None = None
    paper_ids: list[uuid.UUID] = Field(default_factory=list)

    @field_validator("paper_ids")
    @classmethod
    def cap_create_paper_ids(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        return _validate_note_paper_ids(value)


class NoteUpdate(BaseModel):
    title: str | None = None
    summary: str | None = None
    observations: list[str] | None = None
    hypotheses: list[str] | None = None
    questions: list[str] | None = None
    next_steps: list[str] | None = None
    tags: list[str] | None = None
    review_status: ReviewStatus | None = None
    extracted_text: str | None = None
    paper_ids: list[uuid.UUID] | None = None

    @field_validator("paper_ids")
    @classmethod
    def cap_update_paper_ids(
        cls, value: list[uuid.UUID] | None
    ) -> list[uuid.UUID] | None:
        if value is None:
            return None
        return _validate_note_paper_ids(value)


class NoteLinkSource(str, Enum):
    researcher = "researcher"
    ai = "ai"


class ConnectReasonMode(str, Enum):
    snippet = "snippet"
    llm = "llm"


class NotePaperLink(BaseModel):
    paper_id: uuid.UUID
    source: NoteLinkSource = NoteLinkSource.researcher
    similarity: float | None = None
    snippet: str | None = None
    reason: str | None = None
    reason_mode: ConnectReasonMode | None = None


class NoteResponse(ExtractedNote):
    id: uuid.UUID
    source_type: NoteSourceType
    raw_transcript: str
    cleaned_transcript: str
    review_status: ReviewStatus
    model: str | None = None
    prompt_version: str | None = None
    original_filename: str | None = None
    page_count: int | None = None
    extracted_text: str | None = None
    paper_ids: list[uuid.UUID] = Field(default_factory=list)
    paper_links: list[NotePaperLink] = Field(default_factory=list)
    related_generated_at: datetime | None = None
    processing_status: NoteStatus
    processing_error: str | None = None
    chunk_count: int = 0
    revision: int = 1
    sha256: str | None = None
    accession_status: AccessionStatus = AccessionStatus.accessioned
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RelatedPaper(BaseModel):
    paper_id: uuid.UUID
    title: str
    year: int | None = None
    similarity: float
    snippet: str
    reason: str
    reason_mode: ConnectReasonMode
    linked: bool
    source: NoteLinkSource | None = None
    page: int | None = None
    section: str | None = None


class ConnectRequest(BaseModel):
    reason_mode: ConnectReasonMode = ConnectReasonMode.llm
    top_k: int | None = Field(default=None, ge=1, le=8)


class ConnectResponse(BaseModel):
    note_id: uuid.UUID
    reason_mode: ConnectReasonMode
    papers: list[RelatedPaper]
    linked_paper_ids: list[uuid.UUID]
    skipped_paper_ids: list[uuid.UUID] = Field(default_factory=list)
    model: str | None = None
    prompt_version: str
    generated_at: datetime


class NoteChunkResponse(BaseModel):
    id: uuid.UUID
    note_id: uuid.UUID
    chunk_index: int
    text: str
    page: int | None = None
    section: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class VoiceNoteCreate(NoteCreate):
    pass


class VoiceNoteUpdate(BaseModel):
    title: str | None = None
    summary: str | None = None
    observations: list[str] | None = None
    hypotheses: list[str] | None = None
    questions: list[str] | None = None
    next_steps: list[str] | None = None
    tags: list[str] | None = None
    review_status: ReviewStatus | None = None


class VoiceNoteResponse(ExtractedNote):
    id: uuid.UUID
    raw_transcript: str
    cleaned_transcript: str
    review_status: ReviewStatus
    model: str | None = None
    prompt_version: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PaperStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class DigestStatus(str, Enum):
    pending = "pending"
    ready = "ready"
    failed = "failed"


class PaperDigest(BaseModel):
    problem: str = ""
    method: str = ""
    key_results: str = ""
    limitations: str = ""


class PaperUpdate(BaseModel):
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    abstract: str | None = None
    tags: list[str] | None = None


class PaperResponse(BaseModel):
    id: uuid.UUID
    title: str
    authors: list[str]
    year: int | None = None
    abstract: str | None = None
    summary: str | None = None
    digest: PaperDigest = Field(default_factory=PaperDigest)
    digest_status: DigestStatus = DigestStatus.pending
    source: str
    tags: list[str]
    original_filename: str
    page_count: int | None = None
    processing_status: PaperStatus
    processing_error: str | None = None
    chunk_count: int = 0
    revision: int = 1
    sha256: str | None = None
    accession_status: AccessionStatus = AccessionStatus.accessioned
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LibraryUploadResponse(BaseModel):
    kind: Literal["paper", "note", "document"]
    paper: PaperResponse | None = None
    note: NoteResponse | None = None
    document: "LibraryDocumentResponse | None" = None
    suggested_kind: Literal["paper", "note", "document"] | None = None


class SpaceResponse(BaseModel):
    id: uuid.UUID
    slug: str
    name: str


class LibraryRecordResponse(BaseModel):
    id: uuid.UUID
    space_id: uuid.UUID
    content_type: Literal["scholarly_article", "research_note", "document"]
    title: str
    status: str
    updated_at: datetime
    snippet: str | None = None
    highlight: str | None = None
    revision: int = 1
    checksum: str | None = None
    accession_status: str = "accessioned"


class RecordPageResponse(BaseModel):
    items: list[LibraryRecordResponse]
    total: int
    limit: int
    offset: int


class CatalogSearchRequest(BaseModel):
    query: str = ""
    space_id: uuid.UUID | None = None
    content_type: str | None = None
    record_ids: list[uuid.UUID] | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    offset: int | None = Field(default=None, ge=0)


class LibraryDocumentUpdate(BaseModel):
    title: str | None = None


class LibraryDocumentResponse(BaseModel):
    id: uuid.UUID
    title: str
    original_filename: str
    mime_type: str | None = None
    extracted_text: str | None = None
    processing_status: PaperStatus
    processing_error: str | None = None
    chunk_count: int = 0
    revision: int = 1
    sha256: str | None = None
    accession_status: AccessionStatus = AccessionStatus.accessioned
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PaperChunkResponse(BaseModel):
    id: uuid.UUID
    paper_id: uuid.UUID
    chunk_index: int
    text: str
    page: int | None = None
    section: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class CitationSourceType(str, Enum):
    paper = "paper"
    voice = "voice"
    handwritten = "handwritten"
    document = "document"
    web = "web"


class AskCitation(BaseModel):
    index: int
    source_type: CitationSourceType
    source_id: uuid.UUID | None = None
    chunk_id: uuid.UUID | None = None
    title: str
    page: int | None = None
    section: str | None = None
    year: int | None = None
    snippet: str
    similarity: float = 0.0
    url: str | None = None


class QueryKind(str, Enum):
    library = "library"
    field_wide = "field_wide"


class ExternalSearchStatus(str, Enum):
    unavailable = "unavailable"
    ready = "ready"
    ran = "ran"
    failed = "failed"


class LibraryCoverage(BaseModel):
    paper_count: int
    unique_retrieved_papers: int
    years: list[int] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str
    include_papers: bool = True
    include_voice_notes: bool = True
    include_handwritten_notes: bool = True
    top_k: int | None = Field(default=None, ge=1, le=16)
    external_search: bool = False
    paper_ids: list[uuid.UUID] | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question cannot be empty")
        return stripped


class AskResponse(BaseModel):
    answer: str
    insufficient_evidence: bool
    max_similarity: float | None = None
    citations: list[AskCitation]
    model: str
    prompt_version: str
    query_kind: QueryKind = QueryKind.library
    library_coverage: LibraryCoverage
    suggest_external_search: bool = False
    external_search_status: ExternalSearchStatus = ExternalSearchStatus.unavailable
    request_id: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatCitation(BaseModel):
    """Evidence the agent actually read through a tool.

    Mirrors AskCitation, but `similarity` is absent for chunks the agent read
    in source order instead of retrieving by vector search.
    """

    index: int
    source_type: CitationSourceType
    source_id: uuid.UUID
    chunk_id: uuid.UUID
    title: str
    page: int | None = None
    section: str | None = None
    year: int | None = None
    snippet: str
    similarity: float | None = None
    url: str | None = None


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class ChatFiledAttachment(BaseModel):
    kind: Literal["paper", "note", "document"]
    id: uuid.UUID
    title: str
    filename: str
    status: str
    preview: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    include_papers: bool = True
    include_voice_notes: bool = True
    include_handwritten_notes: bool = True
    include_documents: bool = True
    top_k: int | None = Field(default=None, ge=1, le=16)
    conversation_id: uuid.UUID | None = None
    attachments: list[ChatFiledAttachment] = Field(default_factory=list)
    user_id: str | None = None
    space_ids: list[uuid.UUID] | None = None
    space_id: uuid.UUID | None = None
    record_ids: list[uuid.UUID] | None = None
    library_mode: bool | None = None

    @field_validator("messages")
    @classmethod
    def conversation_is_usable(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("Send at least one message")
        if value[-1].role is not ChatRole.user:
            raise ValueError("The last message must come from the user")
        return value

    @model_validator(mode="after")
    def question_or_attachments(self) -> "ChatRequest":
        last = self.messages[-1]
        if last.content.strip() or self.attachments:
            return self
        raise ValueError("Question cannot be empty")

    def is_library_mode(self) -> bool:
        """Collection-scoped turns fail closed; personal chat stays fail-open."""
        if self.library_mode is True:
            return True
        if self.library_mode is False:
            return False
        return self.space_id is not None or bool(self.record_ids)


class ChatEventType(str, Enum):
    tool_call = "tool_call"
    tool_result = "tool_result"
    artifact = "artifact"
    token = "token"
    citations = "citations"
    verdict = "verdict"
    todo = "todo"
    subagent = "subagent"
    compact = "compact"
    attachment = "attachment"
    conversation = "conversation"
    notice = "notice"
    progress = "progress"
    approval = "approval"
    done = "done"
    error = "error"


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    turn_count: int = 0

    model_config = {"from_attributes": True}


class ConversationTurnResponse(BaseModel):
    id: uuid.UUID
    turn_index: int
    payload: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetail(ConversationSummary):
    turns: list[ConversationTurnResponse] = Field(default_factory=list)


class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"


class ChatTodoItem(BaseModel):
    id: str
    content: str
    status: TodoStatus


class SubagentStatus(str, Enum):
    started = "started"
    finished = "finished"
    failed = "failed"


class CompactMode(str, Enum):
    truncate = "truncate"
    summarize = "summarize"


class ArtifactKind(str, Enum):
    """Structured tool output the UI renders instead of reading as prose."""

    comparison = "comparison"
    skill = "skill"
    memory = "memory"


class ChatArtifact(BaseModel):
    kind: ArtifactKind
    tool: str
    data: dict


class GateStatus(str, Enum):
    """How well the evidence gate thinks the answer is backed up.

    `unchecked` means the deterministic checks passed but the LLM reviewer was
    expected and could not be reached, so nobody looked at whether the prose
    actually follows from the snippets. `incomplete` means a distinct part of
    the question was not answered. `impossible` means the library cannot
    satisfy the request, so the loop stops instead of retrying.
    """

    supported = "supported"
    unsupported = "unsupported"
    unchecked = "unchecked"
    retrying = "retrying"
    incomplete = "incomplete"
    impossible = "impossible"


class GateProblemKind(str, Enum):
    fabricated_citation = "fabricated_citation"
    no_evidence_gathered = "no_evidence_gathered"
    uncited_answer = "uncited_answer"
    unsupported_claim = "unsupported_claim"
    judge_unavailable = "judge_unavailable"
    unaddressed_part = "unaddressed_part"
    impossible = "impossible"


class ChatGateProblem(BaseModel):
    kind: GateProblemKind
    detail: str


class ChatVerdict(BaseModel):
    status: GateStatus
    reason: str = ""
    checked_by: str = "deterministic"
    problems: list[ChatGateProblem] = Field(default_factory=list)


DEFAULT_COMPARE_DIMENSIONS = (
    "problem",
    "method",
    "dataset",
    "evaluation",
    "key_results",
    "strengths",
    "limitations",
)
MIN_COMPARE_PAPERS = 2
MAX_COMPARE_PAPERS = 4
MAX_COMPARE_DIMENSIONS = 12
MAX_DIMENSION_LABEL_CHARS = 64


def normalize_paper_ids(value: list[uuid.UUID]) -> list[uuid.UUID]:
    if len(value) < MIN_COMPARE_PAPERS:
        raise ValueError(f"Select at least {MIN_COMPARE_PAPERS} papers")
    if len(value) > MAX_COMPARE_PAPERS:
        raise ValueError(f"Select at most {MAX_COMPARE_PAPERS} papers")
    unique: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for paper_id in value:
        if paper_id in seen:
            continue
        seen.add(paper_id)
        unique.append(paper_id)
    if len(unique) < MIN_COMPARE_PAPERS:
        raise ValueError(f"Select at least {MIN_COMPARE_PAPERS} distinct papers")
    return unique


def normalize_dimensions(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        label = " ".join(raw.split())
        if not label:
            raise ValueError("Dimension labels cannot be blank")
        if len(label) > MAX_DIMENSION_LABEL_CHARS:
            raise ValueError(
                f"Dimension labels must be {MAX_DIMENSION_LABEL_CHARS} characters or fewer"
            )
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(label)
    if not cleaned:
        raise ValueError("Select at least one comparison dimension")
    if len(cleaned) > MAX_COMPARE_DIMENSIONS:
        raise ValueError(
            f"Select at most {MAX_COMPARE_DIMENSIONS} comparison dimensions"
        )
    return cleaned


class CompareRequest(BaseModel):
    paper_ids: list[uuid.UUID]
    dimensions: list[str] = Field(
        default_factory=lambda: list(DEFAULT_COMPARE_DIMENSIONS)
    )


class CompareLinkedNote(BaseModel):
    id: uuid.UUID
    paper_id: uuid.UUID
    title: str
    source_type: CitationSourceType


class ComparePaperResult(BaseModel):
    paper_id: uuid.UUID
    title: str
    year: int | None = None
    values: dict[str, str]
    linked_notes: list[CompareLinkedNote] = Field(default_factory=list)


class CompareSynthesis(BaseModel):
    agreements: str = ""
    disagreements: str = ""
    research_gap: str = ""


class CompareCitation(BaseModel):
    index: int
    source_type: CitationSourceType
    source_id: uuid.UUID
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    title: str
    page: int | None = None
    section: str | None = None
    year: int | None = None
    snippet: str
    similarity: float


class CompareResponse(BaseModel):
    id: uuid.UUID
    paper_ids: list[uuid.UUID]
    dimensions: list[str]
    papers: list[ComparePaperResult]
    synthesis: CompareSynthesis
    citations: list[CompareCitation]
    model: str
    prompt_version: str
    created_at: datetime


class CompareSummary(BaseModel):
    id: uuid.UUID
    paper_ids: list[uuid.UUID]
    paper_titles: list[str]
    created_at: datetime


class MemoryCategory(str, Enum):
    preference = "preference"
    hypothesis = "hypothesis"
    focus = "focus"
    workflow = "workflow"
    other = "other"


class MemoryResponse(BaseModel):
    id: uuid.UUID
    key: str
    content: str
    category: str
    source_turn: str | None = None
    created_at: datetime
    updated_at: datetime
