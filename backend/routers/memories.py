"""List and delete cross-session agent memories (Settings oversight)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from schemas import MemoryResponse
from services.agent.memory import (
    MemoryError,
    delete_memory,
    memory_as_dict,
    search_memories,
)

router = APIRouter(prefix="/api/memories", tags=["memories"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _to_response(row) -> MemoryResponse:
    data = memory_as_dict(row)
    return MemoryResponse(
        id=row.id,
        key=data["key"],
        content=data["content"],
        category=data["category"],
        source_turn=data["source_turn"],
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("", response_model=list[MemoryResponse])
async def list_memories(
    session: SessionDep,
    query: str | None = None,
    category: str | None = None,
) -> list[MemoryResponse]:
    try:
        rows = await search_memories(session, query=query, category=category)
    except MemoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [_to_response(row) for row in rows]


@router.delete("/{key}", status_code=204)
async def remove_memory(key: str, session: SessionDep) -> None:
    try:
        deleted = await delete_memory(session, key=key)
    except MemoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No memory with key={key!r}")
