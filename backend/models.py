import enum
import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 768
DEFAULT_SPACE_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
DEFAULT_SPACE_SLUG = "default"
DEFAULT_SPACE_ID_SQL = text("'00000000-0000-4000-8000-000000000001'")


class Base(DeclarativeBase):
    pass


class Space(Base):
    """A library partition. Stub ACL in Phase 1; not an org/tenant model."""

    __tablename__ = "spaces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    slug: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


class SpaceMembership(Base):
    __tablename__ = "space_memberships"
    __table_args__ = (UniqueConstraint("space_id", "user_id"),)

    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("spaces.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


class ReviewStatus(str, enum.Enum):
    generated = "generated"
    draft = "draft"
    reviewed = "reviewed"
    accepted = "accepted"


class NoteSourceType(str, enum.Enum):
    voice = "voice"
    handwritten = "handwritten"


class ProcessingStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class AccessionStatus(str, enum.Enum):
    """Whether the original may be searched.

    received: bytes are stored, not yet in search/RAG.
    accessioned: parse produced text; catalog and Ask may use it.
    rejected: blank, failed, or refused. Not searchable.
    """

    received = "received"
    accessioned = "accessioned"
    rejected = "rejected"


ACCESSIONED_DEFAULT_SQL = text("'accessioned'")


class NoteLinkSource(str, enum.Enum):
    researcher = "researcher"
    ai = "ai"


class ConnectReasonMode(str, enum.Enum):
    snippet = "snippet"
    llm = "llm"


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("spaces.id"),
        nullable=False,
        index=True,
        default=DEFAULT_SPACE_ID,
        server_default=DEFAULT_SPACE_ID_SQL,
    )
    source_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default=NoteSourceType.voice.value
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    observations: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    hypotheses: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    questions: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    next_steps: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    raw_transcript: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cleaned_transcript: Mapped[str] = mapped_column(Text, nullable=False, default="")
    review_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ReviewStatus.generated.value
    )
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    original_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ProcessingStatus.pending.value
    )
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    accession_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AccessionStatus.accessioned.value,
        server_default=ACCESSIONED_DEFAULT_SQL,
    )
    related_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    chunks: Mapped[list["NoteChunk"]] = relationship(
        back_populates="note", cascade="all, delete-orphan"
    )
    paper_links: Mapped[list["NotePaper"]] = relationship(
        back_populates="note", cascade="all, delete-orphan"
    )


class NoteChunk(Base):
    __tablename__ = "note_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    note_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("notes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(text, ''))", persisted=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    note: Mapped[Note] = relationship(back_populates="chunks")


class PaperStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class DigestStatus(str, enum.Enum):
    pending = "pending"
    ready = "ready"
    failed = "failed"


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("spaces.id"),
        nullable=False,
        index=True,
        default=DEFAULT_SPACE_ID,
        server_default=DEFAULT_SPACE_ID_SQL,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    authors: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    digest: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    digest_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DigestStatus.pending.value
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="upload")
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_file: Mapped[str] = mapped_column(String(1024), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processing_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=PaperStatus.pending.value
    )
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    accession_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AccessionStatus.accessioned.value,
        server_default=ACCESSIONED_DEFAULT_SQL,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    chunks: Mapped[list["PaperChunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    note_links: Mapped[list["NotePaper"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(text, ''))", persisted=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    paper: Mapped[Paper] = relationship(back_populates="chunks")


class NotePaper(Base):
    """A link between a note and a paper (researcher or AI-suggested)."""

    __tablename__ = "note_papers"

    note_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("notes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default=NoteLinkSource.researcher.value
    )
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    note: Mapped[Note] = relationship(back_populates="paper_links")
    paper: Mapped[Paper] = relationship(back_populates="note_links")


class NotePaperSkip(Base):
    """An AI-suggested pair the researcher dismissed; auto-link will not re-add it."""

    __tablename__ = "note_paper_skips"

    note_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("notes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


class PaperComparison(Base):
    __tablename__ = "paper_comparisons"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    paper_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    dimensions: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    result: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    citations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


class LibraryDocument(Base):
    """A generic library file (markdown, csv, docx, or a PDF filed as a document)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("spaces.id"),
        nullable=False,
        index=True,
        default=DEFAULT_SPACE_ID,
        server_default=DEFAULT_SPACE_ID_SQL,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_file: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ProcessingStatus.pending.value
    )
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    accession_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AccessionStatus.accessioned.value,
        server_default=ACCESSIONED_DEFAULT_SQL,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    chunks: Mapped[list["LibraryDocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class LibraryDocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(text, ''))", persisted=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    document: Mapped[LibraryDocument] = relationship(back_populates="chunks")


class Conversation(Base):
    """A persisted chat thread the researcher can reopen from history."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="New chat")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    turns: Mapped[list["ConversationTurn"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationTurn.turn_index",
    )


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    conversation: Mapped[Conversation] = relationship(back_populates="turns")


class AgentMemory(Base):
    """Cross-session preferences and hypotheses the chat agent can recall."""

    __tablename__ = "agent_memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    key: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="other")
    source_turn: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Recall used to be substring counting, which misses "I care about
    # retrieval" for a question about "search quality". Nullable because the
    # embedding service can be down when a memory is written, and a memory
    # without a vector must still be stored and still be findable by keyword.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    # Lets a record that nothing has needed for months sink in the ranking
    # instead of holding a slot in the recall window forever.
    last_recalled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


class WorkflowStep(Base):
    """One journalled unit of work inside a deterministic workflow.

    A four-paper comparison is four model calls plus a synthesis. Without a
    journal, a failure on the fourth throws away the first three and the
    researcher pays for them again. With one, a retry under the same `run_id`
    replays the finished steps from here and only recomputes what is missing.

    `step_key` is a hash of what the step *means* (kind, label, prompt,
    schema), never its position or completion order -- parallel steps finish in
    an unpredictable order, so an index-based key would replay the wrong result
    into the wrong slot.
    """

    __tablename__ = "workflow_steps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    step_key: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    label: Mapped[str] = mapped_column(String(240), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="done")
    result: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        UniqueConstraint("run_id", "step_key", name="uq_workflow_steps_run_key"),
    )
