"""Add paper summary, digest, and digest_status for post-ingest understanding.

Revision ID: 011_paper_digests
Revises: 010_documents_conversations
Create Date: 2026-09-09

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "011_paper_digests"
down_revision: str | Sequence[str] | None = "010_documents_conversations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column(
        "papers",
        sa.Column(
            "digest",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        "papers",
        sa.Column(
            "digest_status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
    )
    op.create_index(
        "ix_papers_digest_status", "papers", ["digest_status"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_papers_digest_status", table_name="papers")
    op.drop_column("papers", "digest_status")
    op.drop_column("papers", "digest")
    op.drop_column("papers", "summary")
