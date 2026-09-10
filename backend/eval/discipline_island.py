"""Un-accessioned and blank-original fixtures for the discipline regression suite.

The received paper has chunks and a unique token so a miss proves the
accession filter, not a missing index. The blank paper has no extractable
text; Ask must abstain.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.fixtures import axis
from models import (
    DEFAULT_SPACE_ID,
    AccessionStatus,
    Paper,
    PaperChunk,
    PaperStatus,
    Space,
    SpaceMembership,
)

DISCIPLINE_TAG = "discipline-fixture"
FAR_AXIS = 41

RECEIVED_PAPER_ID = uuid.UUID("00000000-0000-4000-8000-000000000031")
BLANK_PAPER_ID = uuid.UUID("00000000-0000-4000-8000-000000000032")

RECEIVED_TOKEN = "ZXQ-RECV-HOLD"
BLANK_TOKEN = "ZXQ-BLANK-OCR"
RECEIVED_TITLE = f"Hold for accession {RECEIVED_TOKEN}"
BLANK_TITLE = f"Unscanned original {BLANK_TOKEN}"
RECEIVED_TEXT = (
    f"{RECEIVED_TITLE}. Document number {RECEIVED_TOKEN} sits in received "
    "status with chunks, so retrieval must ignore it until accession."
)
BLANK_TEXT = f"{BLANK_TITLE}. No extractable text."


async def cleanup_discipline_island(session: AsyncSession) -> None:
    papers = (
        (
            await session.execute(
                select(Paper).where(Paper.tags.contains([DISCIPLINE_TAG]))
            )
        )
        .scalars()
        .all()
    )
    for paper in papers:
        await session.delete(paper)
    await session.commit()


async def seed_discipline_island(session: AsyncSession) -> None:
    await cleanup_discipline_island(session)
    now = datetime.now(UTC)
    if await session.get(Space, DEFAULT_SPACE_ID) is None:
        session.add(Space(id=DEFAULT_SPACE_ID, slug="default", name="Default library"))
        await session.flush()
    if await session.get(SpaceMembership, (DEFAULT_SPACE_ID, "alice")) is None:
        session.add(SpaceMembership(space_id=DEFAULT_SPACE_ID, user_id="alice"))

    far = axis(FAR_AXIS)
    received = Paper(
        id=RECEIVED_PAPER_ID,
        space_id=DEFAULT_SPACE_ID,
        title=RECEIVED_TITLE,
        authors=["Eval Fixture"],
        year=2023,
        abstract=RECEIVED_TEXT[:400],
        tags=[DISCIPLINE_TAG],
        original_filename="received.pdf",
        original_file="/tmp/eval-discipline-received.pdf",
        processing_status=PaperStatus.ready.value,
        revision=1,
        sha256="a" * 64,
        accession_status=AccessionStatus.received.value,
        created_at=now,
        updated_at=now,
    )
    blank = Paper(
        id=BLANK_PAPER_ID,
        space_id=DEFAULT_SPACE_ID,
        title=BLANK_TITLE,
        authors=["Eval Fixture"],
        year=2023,
        abstract=BLANK_TEXT[:400],
        tags=[DISCIPLINE_TAG],
        original_filename="blank.pdf",
        original_file="/tmp/eval-discipline-blank.pdf",
        processing_status=PaperStatus.failed.value,
        processing_error="No extractable text (blank or unscanned original).",
        revision=1,
        sha256="b" * 64,
        accession_status=AccessionStatus.rejected.value,
        created_at=now,
        updated_at=now,
    )
    session.add(received)
    session.add(blank)
    await session.flush()
    session.add(
        PaperChunk(
            paper_id=RECEIVED_PAPER_ID,
            chunk_index=0,
            text=RECEIVED_TEXT,
            page=1,
            section="Body",
            embedding=far,
        )
    )
    await session.commit()
