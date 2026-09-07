"""Add agent_memories for cross-session chat agent recall.

Revision ID: 005_agent_memories
Revises: 004_paper_comparisons
Create Date: 2026-09-05

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005_agent_memories"
down_revision: str | Sequence[str] | None = "004_paper_comparisons"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key", sa.String(length=120), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("source_turn", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    op.create_index(
        "ix_agent_memories_updated_at",
        "agent_memories",
        ["updated_at"],
        unique=False,
    )
    op.create_index(
        "ix_agent_memories_category",
        "agent_memories",
        ["category"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_agent_memories_category", table_name="agent_memories")
    op.drop_index("ix_agent_memories_updated_at", table_name="agent_memories")
    op.drop_table("agent_memories")
