"""Cross-session researcher memory stored in Postgres.

The chat agent reads and writes these via tools. Settings can list and delete
them. Keys are short stable strings the model reuses across turns.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import AgentMemory

MEMORY_CATEGORIES = frozenset(
    {"preference", "hypothesis", "focus", "workflow", "other"}
)
DEFAULT_CATEGORY = "other"
SEARCH_LIMIT = 20


class MemoryError(ValueError):
    """Raised when a memory tool argument is invalid."""


def _normalize_category(value: object | None) -> str:
    if value is None or str(value).strip() == "":
        return DEFAULT_CATEGORY
    category = str(value).strip().lower()
    if category not in MEMORY_CATEGORIES:
        raise MemoryError(
            f"category must be one of {sorted(MEMORY_CATEGORIES)}, got {value!r}"
        )
    return category


def _normalize_key(value: object) -> str:
    key = str(value or "").strip()
    if not key:
        raise MemoryError("key is required and cannot be empty")
    if len(key) > 120:
        raise MemoryError("key must be at most 120 characters")
    return key


def memory_as_dict(row: AgentMemory) -> dict:
    return {
        "id": str(row.id),
        "key": row.key,
        "content": row.content,
        "category": row.category,
        "source_turn": row.source_turn,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def search_memories(
    session: AsyncSession,
    *,
    query: str | None = None,
    category: str | None = None,
    limit: int = SEARCH_LIMIT,
) -> list[AgentMemory]:
    stmt = select(AgentMemory).order_by(AgentMemory.updated_at.desc())
    if category:
        stmt = stmt.where(AgentMemory.category == _normalize_category(category))
    q = (query or "").strip()
    if q:
        pattern = f"%{q}%"
        stmt = stmt.where(
            or_(AgentMemory.key.ilike(pattern), AgentMemory.content.ilike(pattern))
        )
    stmt = stmt.limit(max(1, min(limit, 50)))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def upsert_memory(
    session: AsyncSession,
    *,
    key: object,
    content: object,
    category: object | None = None,
    source_turn: object | None = None,
) -> AgentMemory:
    normalized_key = _normalize_key(key)
    text = str(content or "").strip()
    if not text:
        raise MemoryError("content is required and cannot be empty")
    cat = _normalize_category(category)
    note = str(source_turn).strip() if source_turn else None
    if note == "":
        note = None

    result = await session.execute(
        select(AgentMemory).where(AgentMemory.key == normalized_key)
    )
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        row = AgentMemory(
            key=normalized_key,
            content=text,
            category=cat,
            source_turn=note,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
    else:
        row.content = text
        row.category = cat
        if note is not None:
            row.source_turn = note
        row.updated_at = now
    await session.commit()
    await session.refresh(row)
    return row


async def delete_memory(session: AsyncSession, *, key: object) -> bool:
    normalized_key = _normalize_key(key)
    result = await session.execute(
        select(AgentMemory).where(AgentMemory.key == normalized_key)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return False
    await session.delete(row)
    await session.commit()
    return True


async def get_memory_by_key(session: AsyncSession, key: str) -> AgentMemory | None:
    result = await session.execute(select(AgentMemory).where(AgentMemory.key == key))
    return result.scalar_one_or_none()
