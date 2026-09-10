"""Add revision, checksum, and accession status on library records.

Revision ID: 015_accession
Revises: 014_spaces_acl
Create Date: 2026-09-10

The original is the preservation copy: it is not overwritten, and it is not
searchable until accessioned. Existing ready rows backfill as accessioned so
current tests and the eval library keep retrieving. Failed rows become
rejected; anything still in the pipeline is received.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "015_accession"
down_revision: str | Sequence[str] | None = "014_spaces_acl"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("papers", "notes", "documents")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(
            table,
            sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        )
        op.add_column(table, sa.Column("sha256", sa.String(length=64), nullable=True))
        op.add_column(
            table,
            sa.Column(
                "accession_status",
                sa.String(length=32),
                nullable=False,
                server_default="accessioned",
            ),
        )
        op.execute(
            sa.text(
                f"UPDATE {table} SET accession_status = 'rejected' "
                f"WHERE processing_status = 'failed'"
            )
        )
        op.execute(
            sa.text(
                f"UPDATE {table} SET accession_status = 'received' "
                f"WHERE processing_status NOT IN ('ready', 'failed')"
            )
        )


def downgrade() -> None:
    for table in TABLES:
        op.drop_column(table, "accession_status")
        op.drop_column(table, "sha256")
        op.drop_column(table, "revision")
