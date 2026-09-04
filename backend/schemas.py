import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


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


class NoteCreate(ExtractedNote):
    raw_transcript: str
    cleaned_transcript: str
    review_status: ReviewStatus = ReviewStatus.generated
    model: str | None = None
    prompt_version: str | None = None


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
    paper_id: uuid.UUID | None = None


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
    paper_id: uuid.UUID | None = None
    processing_status: NoteStatus
    processing_error: str | None = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


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
    source: str
    tags: list[str]
    original_filename: str
    page_count: int | None = None
    processing_status: PaperStatus
    processing_error: str | None = None
    chunk_count: int = 0
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


class AskCitation(BaseModel):
    index: int
    source_type: CitationSourceType
    source_id: uuid.UUID
    chunk_id: uuid.UUID
    title: str
    page: int | None = None
    section: str | None = None
    year: int | None = None
    snippet: str
    similarity: float


class QueryKind(str, Enum):
    library = "library"
    field_wide = "field_wide"


class ExternalSearchStatus(str, Enum):
    unavailable = "unavailable"


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
