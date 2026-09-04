"""Add notes and note_chunks; migrate voice_notes.

Revision ID: 003_notes
Revises: 002_papers
Create Date: 2026-09-04

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "003_notes"
down_revision: str | Sequence[str] | None = "002_papers"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "notes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "source_type",
            sa.String(length=32),
            nullable=False,
            server_default="voice",
        ),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "observations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "hypotheses",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "questions",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "next_steps",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("raw_transcript", sa.Text(), nullable=False, server_default=""),
        sa.Column("cleaned_transcript", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "review_status",
            sa.String(length=32),
            nullable=False,
            server_default="generated",
        ),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("original_filename", sa.String(length=512), nullable=True),
        sa.Column("original_file", sa.String(length=1024), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "processing_status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("processing_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_notes_created_at", "notes", ["created_at"], unique=False)
    op.create_index("ix_notes_source_type", "notes", ["source_type"], unique=False)
    op.create_index("ix_notes_paper_id", "notes", ["paper_id"], unique=False)

    op.execute(
        """
        INSERT INTO notes (
            id, source_type, title, summary, observations, hypotheses,
            questions, next_steps, tags, raw_transcript, cleaned_transcript,
            review_status, model, prompt_version, processing_status,
            created_at, updated_at
        )
        SELECT
            id, 'voice', title, summary, observations, hypotheses,
            questions, next_steps, tags, raw_transcript, cleaned_transcript,
            review_status, model, prompt_version, 'pending',
            created_at, updated_at
        FROM voice_notes
        """
    )

    op.create_table(
        "note_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("note_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("section", sa.String(length=512), nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column(
            "extra",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["note_id"], ["notes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_note_chunks_note_id", "note_chunks", ["note_id"], unique=False)
    op.execute(
        "CREATE INDEX ix_note_chunks_embedding ON note_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.drop_index("ix_voice_notes_created_at", table_name="voice_notes")
    op.drop_table("voice_notes")


def downgrade() -> None:
    op.create_table(
        "voice_notes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column(
            "observations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "hypotheses",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "questions",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "next_steps",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("raw_transcript", sa.Text(), nullable=False),
        sa.Column("cleaned_transcript", sa.Text(), nullable=False),
        sa.Column(
            "review_status",
            sa.String(length=32),
            nullable=False,
            server_default="generated",
        ),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_voice_notes_created_at", "voice_notes", ["created_at"], unique=False
    )
    op.execute(
        """
        INSERT INTO voice_notes (
            id, title, summary, observations, hypotheses, questions,
            next_steps, tags, raw_transcript, cleaned_transcript,
            review_status, model, prompt_version, created_at, updated_at
        )
        SELECT
            id, title, summary, observations, hypotheses, questions,
            next_steps, tags, raw_transcript, cleaned_transcript,
            review_status, model, prompt_version, created_at, updated_at
        FROM notes
        WHERE source_type = 'voice'
        """
    )

    op.execute("DROP INDEX IF EXISTS ix_note_chunks_embedding")
    op.drop_index("ix_note_chunks_note_id", table_name="note_chunks")
    op.drop_table("note_chunks")
    op.drop_index("ix_notes_paper_id", table_name="notes")
    op.drop_index("ix_notes_source_type", table_name="notes")
    op.drop_index("ix_notes_created_at", table_name="notes")
    op.drop_table("notes")
