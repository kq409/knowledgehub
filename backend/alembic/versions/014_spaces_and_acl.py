"""Add spaces, memberships, and space_id on library records.

Revision ID: 014_spaces_acl
Revises: 013_workflow_steps
Create Date: 2026-09-10

Phase 1 stub ACL: every paper/note/document belongs to a space. Existing
rows backfill into the default space. Alice and bob are both members of
default so current tests and the eval harness keep seeing the library.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "014_spaces_acl"
down_revision: str | Sequence[str] | None = "013_workflow_steps"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "spaces",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("slug"),
    )
    op.create_table(
        "space_memberships",
        sa.Column("space_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["space_id"], ["spaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("space_id", "user_id"),
    )
    op.create_index("ix_space_memberships_user_id", "space_memberships", ["user_id"])

    op.execute(
        sa.text(
            """
            INSERT INTO spaces (id, slug, name, created_at)
            VALUES (
                '00000000-0000-4000-8000-000000000001'::uuid,
                'default',
                'Default library',
                NOW()
            )
            """
        )
    )
    op.execute(
        sa.text(
            """
            INSERT INTO space_memberships (space_id, user_id, created_at)
            VALUES
                ('00000000-0000-4000-8000-000000000001'::uuid, 'alice', NOW()),
                ('00000000-0000-4000-8000-000000000001'::uuid, 'bob', NOW())
            """
        )
    )

    for table in ("papers", "notes", "documents"):
        op.add_column(
            table,
            sa.Column(
                "space_id",
                postgresql.UUID(as_uuid=True),
                nullable=True,
            ),
        )
        op.execute(
            sa.text(
                f"UPDATE {table} SET space_id = "
                "'00000000-0000-4000-8000-000000000001'::uuid "
                "WHERE space_id IS NULL"
            )
        )
        op.alter_column(table, "space_id", nullable=False)
        op.create_index(f"ix_{table}_space_id", table, ["space_id"])
        op.create_foreign_key(
            f"fk_{table}_space_id",
            table,
            "spaces",
            ["space_id"],
            ["id"],
        )


def downgrade() -> None:
    for table in ("documents", "notes", "papers"):
        op.drop_constraint(f"fk_{table}_space_id", table, type_="foreignkey")
        op.drop_index(f"ix_{table}_space_id", table_name=table)
        op.drop_column(table, "space_id")
    op.drop_index("ix_space_memberships_user_id", table_name="space_memberships")
    op.drop_table("space_memberships")
    op.drop_table("spaces")
