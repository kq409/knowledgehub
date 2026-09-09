"""Add memory embeddings and recall recency for vector-based recall.

Revision ID: 012_memory_embeddings
Revises: 011_paper_digests
Create Date: 2026-09-09

Both columns are nullable: memories written while the embedding service is
unreachable must still be stored, and every row that already exists predates
this feature. Recall falls back to keyword matching for rows with no vector.

No ANN index. The store is a handful of rows per researcher and the recall
query already reads them all for the catalog, so an ivfflat index would cost
maintenance and buy nothing at this size.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "012_memory_embeddings"
down_revision: str | Sequence[str] | None = "011_paper_digests"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.add_column(
        "agent_memories",
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
    )
    op.add_column(
        "agent_memories",
        sa.Column("last_recalled_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_memories", "last_recalled_at")
    op.drop_column("agent_memories", "embedding")
