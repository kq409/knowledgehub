"""Stored tsvector columns for hybrid lexical search.

Revision ID: 009_chunk_tsv
Revises: 008_hybrid_search
Create Date: 2026-09-07

"""

from collections.abc import Sequence

from alembic import op

revision: str = "009_chunk_tsv"
down_revision: str | Sequence[str] | None = "008_hybrid_search"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE paper_chunks
        ADD COLUMN IF NOT EXISTS tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_chunks_tsv ON paper_chunks USING GIN (tsv)"
    )
    op.execute(
        """
        ALTER TABLE note_chunks
        ADD COLUMN IF NOT EXISTS tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_note_chunks_tsv ON note_chunks USING GIN (tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_note_chunks_tsv")
    op.execute("DROP INDEX IF EXISTS ix_paper_chunks_tsv")
    op.execute("ALTER TABLE note_chunks DROP COLUMN IF EXISTS tsv")
    op.execute("ALTER TABLE paper_chunks DROP COLUMN IF EXISTS tsv")
