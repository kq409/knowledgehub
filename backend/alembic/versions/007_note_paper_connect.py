"""AI-suggested note↔paper links, skips, and last related-papers run.

Revision ID: 007_note_paper_connect
Revises: 006_note_papers
Create Date: 2026-09-05

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "007_note_paper_connect"
down_revision: str | Sequence[str] | None = "006_note_papers"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "notes",
        sa.Column(
            "related_result", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
    )
    op.add_column(
        "note_papers",
        sa.Column(
            "source",
            sa.String(length=32),
            nullable=False,
            server_default="researcher",
        ),
    )
    op.add_column("note_papers", sa.Column("similarity", sa.Float(), nullable=True))
    op.add_column("note_papers", sa.Column("snippet", sa.Text(), nullable=True))
    op.add_column("note_papers", sa.Column("reason", sa.Text(), nullable=True))
    op.add_column(
        "note_papers",
        sa.Column("reason_mode", sa.String(length=32), nullable=True),
    )
    op.create_table(
        "note_paper_skips",
        sa.Column("note_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["note_id"], ["notes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("note_id", "paper_id"),
    )
    op.create_index(
        "ix_note_paper_skips_paper_id",
        "note_paper_skips",
        ["paper_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_note_paper_skips_paper_id", table_name="note_paper_skips")
    op.drop_table("note_paper_skips")
    op.drop_column("note_papers", "reason_mode")
    op.drop_column("note_papers", "reason")
    op.drop_column("note_papers", "snippet")
    op.drop_column("note_papers", "similarity")
    op.drop_column("note_papers", "source")
    op.drop_column("notes", "related_result")
