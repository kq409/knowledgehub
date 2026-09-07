"""GIN FTS (simple) and trigram indexes for hybrid retrieval.

Revision ID: 008_hybrid_search
Revises: 007_note_paper_connect
Create Date: 2026-09-07

"""

from collections.abc import Sequence

from alembic import op

revision: str = "008_hybrid_search"
down_revision: str | Sequence[str] | None = "007_note_paper_connect"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FTS = "to_tsvector('simple'::regconfig, COALESCE(%s, ''))"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_paper_chunks_fts "
        f"ON paper_chunks USING GIN ({FTS % 'text'})"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_note_chunks_fts "
        f"ON note_chunks USING GIN ({FTS % 'text'})"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_papers_title_fts "
        f"ON papers USING GIN ({FTS % 'title'})"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_notes_title_fts "
        f"ON notes USING GIN ({FTS % 'title'})"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_chunks_text_trgm "
        "ON paper_chunks USING GIN (text gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_note_chunks_text_trgm "
        "ON note_chunks USING GIN (text gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_papers_title_trgm "
        "ON papers USING GIN (title gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_notes_title_trgm "
        "ON notes USING GIN (title gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_notes_title_trgm")
    op.execute("DROP INDEX IF EXISTS ix_papers_title_trgm")
    op.execute("DROP INDEX IF EXISTS ix_note_chunks_text_trgm")
    op.execute("DROP INDEX IF EXISTS ix_paper_chunks_text_trgm")
    op.execute("DROP INDEX IF EXISTS ix_notes_title_fts")
    op.execute("DROP INDEX IF EXISTS ix_papers_title_fts")
    op.execute("DROP INDEX IF EXISTS ix_note_chunks_fts")
    op.execute("DROP INDEX IF EXISTS ix_paper_chunks_fts")
