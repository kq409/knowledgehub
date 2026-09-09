"""Add generic documents, document chunks, and persisted conversations.

Revision ID: 010_documents_conversations
Revises: 009_chunk_tsv
Create Date: 2026-09-08

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "010_documents_conversations"
down_revision: str | Sequence[str] | None = "009_chunk_tsv"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FTS = "to_tsvector('simple'::regconfig, COALESCE(%s, ''))"


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("original_file", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column(
            "processing_status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("processing_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_documents_created_at", "documents", ["created_at"], unique=False
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
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
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
        unique=False,
    )
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        """
        ALTER TABLE document_chunks
        ADD COLUMN IF NOT EXISTS tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_tsv "
        "ON document_chunks USING GIN (tsv)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_document_chunks_fts "
        f"ON document_chunks USING GIN ({FTS % 'text'})"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_documents_title_fts "
        f"ON documents USING GIN ({FTS % 'title'})"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_text_trgm "
        "ON document_chunks USING GIN (text gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_title_trgm "
        "ON documents USING GIN (title gin_trgm_ops)"
    )

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversations_updated_at",
        "conversations",
        ["updated_at"],
        unique=False,
    )

    op.create_table(
        "conversation_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversation_turns_conversation_id",
        "conversation_turns",
        ["conversation_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_turns_conversation_id", table_name="conversation_turns"
    )
    op.drop_table("conversation_turns")
    op.drop_index("ix_conversations_updated_at", table_name="conversations")
    op.drop_table("conversations")

    op.execute("DROP INDEX IF EXISTS ix_documents_title_trgm")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_text_trgm")
    op.execute("DROP INDEX IF EXISTS ix_documents_title_fts")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_fts")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_tsv")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index("ix_documents_created_at", table_name="documents")
    op.drop_table("documents")
