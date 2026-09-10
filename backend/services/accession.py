"""Accession rules: originals stay put; search only sees accessioned records.

The catalog UI may list received items so a researcher can see what is still
in the pipeline. `search_library` and `POST /api/search` do not.
"""

from __future__ import annotations

from datetime import UTC, datetime

from models import AccessionStatus

EMPTY_ORIGINAL = "No extractable text (blank or unscanned original)."
READY = "ready"
FAILED = "failed"


def accessioned_clause(column):
    return column == AccessionStatus.accessioned.value


def is_accessioned(entity) -> bool:
    status = getattr(entity, "accession_status", None)
    return status == AccessionStatus.accessioned.value


def apply_processing_outcome(
    entity,
    *,
    chunk_count: int,
    failed: bool = False,
    error: str | None = None,
) -> None:
    """Set processing + accession together after a parse (or a parse failure)."""
    if failed:
        entity.processing_status = FAILED
        entity.accession_status = AccessionStatus.rejected.value
        entity.processing_error = error
        entity.updated_at = datetime.now(UTC)
        return
    if chunk_count <= 0:
        entity.processing_status = FAILED
        entity.accession_status = AccessionStatus.rejected.value
        entity.processing_error = EMPTY_ORIGINAL
        entity.updated_at = datetime.now(UTC)
        return
    entity.processing_status = READY
    entity.accession_status = AccessionStatus.accessioned.value
    entity.processing_error = None
    entity.updated_at = datetime.now(UTC)


def mark_processing_fields(entity, status: str, error: str | None = None) -> None:
    entity.processing_status = status
    entity.processing_error = error
    entity.updated_at = datetime.now(UTC)
    if status == FAILED:
        entity.accession_status = AccessionStatus.rejected.value
