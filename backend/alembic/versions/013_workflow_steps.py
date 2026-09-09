"""Add workflow_steps so a long multi-call workflow can resume after a failure.

Revision ID: 013_workflow_steps
Revises: 012_memory_embeddings
Create Date: 2026-09-09

The unique constraint on (run_id, step_key) is what makes resume correct: a
replayed run recomputes a step only when its semantic key is absent, and can
never end up with two journal rows claiming the same step.

Postgres rather than files because the journal has to be readable by whichever
replica picks the retry up, and this app already has a database.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "013_workflow_steps"
down_revision: str | Sequence[str] | None = "012_memory_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_steps",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("run_id", sa.String(length=120), nullable=False),
        sa.Column("step_key", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=48), nullable=False),
        sa.Column("label", sa.String(length=240), nullable=False, server_default=""),
        sa.Column(
            "status", sa.String(length=16), nullable=False, server_default="done"
        ),
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "step_key", name="uq_workflow_steps_run_key"),
    )
    op.create_index("ix_workflow_steps_run_id", "workflow_steps", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_workflow_steps_run_id", table_name="workflow_steps")
    op.drop_table("workflow_steps")
