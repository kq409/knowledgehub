"""Replace notes.paper_id with a many-to-many note_papers table.

Revision ID: 006_note_papers
Revises: 005_agent_memories
Create Date: 2026-09-05

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "006_note_papers"
down_revision: str | Sequence[str] | None = "005_agent_memories"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "note_papers",
        sa.Column("note_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["note_id"], ["notes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("note_id", "paper_id"),
    )
    op.create_index(
        "ix_note_papers_paper_id", "note_papers", ["paper_id"], unique=False
    )
    op.execute(
        """
        INSERT INTO note_papers (note_id, paper_id, created_at)
        SELECT id, paper_id, created_at
        FROM notes
        WHERE paper_id IS NOT NULL
        """
    )
    op.drop_index("ix_notes_paper_id", table_name="notes")
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for fk in inspector.get_foreign_keys("notes"):
        if fk.get("constrained_columns") == ["paper_id"] and fk.get("name"):
            op.drop_constraint(fk["name"], "notes", type_="foreignkey")
    op.drop_column("notes", "paper_id")


def downgrade() -> None:
    op.add_column(
        "notes",
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "notes_paper_id_fkey",
        "notes",
        "papers",
        ["paper_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_notes_paper_id", "notes", ["paper_id"], unique=False)
    op.execute(
        """
        UPDATE notes SET paper_id = (
            SELECT paper_id FROM note_papers
            WHERE note_papers.note_id = notes.id
            ORDER BY created_at ASC
            LIMIT 1
        )
        """
    )
    op.drop_index("ix_note_papers_paper_id", table_name="note_papers")
    op.drop_table("note_papers")
