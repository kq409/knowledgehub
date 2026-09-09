from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db import get_session
from models import Conversation, ConversationTurn
from schemas import ConversationDetail, ConversationSummary, ConversationTurnResponse

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]

MAX_TITLE_CHARS = 80


def conversation_title_from(text: str) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "New chat"
    if len(cleaned) <= MAX_TITLE_CHARS:
        return cleaned
    return cleaned[: MAX_TITLE_CHARS - 1].rstrip() + "…"


def _summary(row: Conversation, turn_count: int) -> ConversationSummary:
    return ConversationSummary(
        id=row.id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
        turn_count=turn_count,
    )


async def ensure_conversation(
    session: AsyncSession,
    conversation_id: uuid.UUID | None,
    *,
    first_question: str,
) -> Conversation:
    if conversation_id is not None:
        row = await session.get(Conversation, conversation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return row
    row = Conversation(title=conversation_title_from(first_question))
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def append_turn(
    session: AsyncSession,
    conversation: Conversation,
    payload: dict,
) -> ConversationTurn:
    count = await session.scalar(
        select(func.count(ConversationTurn.id)).where(
            ConversationTurn.conversation_id == conversation.id
        )
    )
    turn = ConversationTurn(
        conversation_id=conversation.id,
        turn_index=int(count or 0),
        payload=payload,
    )
    conversation.updated_at = datetime.now(UTC)
    if conversation.title == "New chat":
        question = str(payload.get("question") or "").strip()
        if question:
            conversation.title = conversation_title_from(question)
    session.add(turn)
    await session.commit()
    await session.refresh(turn)
    return turn


@router.get("", response_model=list[ConversationSummary])
@router.get("/", response_model=list[ConversationSummary], include_in_schema=False)
async def list_conversations(session: SessionDep):
    turn_count = (
        select(func.count(ConversationTurn.id))
        .where(ConversationTurn.conversation_id == Conversation.id)
        .correlate(Conversation)
        .scalar_subquery()
    )
    result = await session.execute(
        select(Conversation, turn_count).order_by(Conversation.updated_at.desc())
    )
    return [_summary(row, int(count or 0)) for row, count in result.all()]


@router.post("", response_model=ConversationSummary, status_code=201)
@router.post(
    "/", response_model=ConversationSummary, status_code=201, include_in_schema=False
)
async def create_conversation(session: SessionDep):
    row = Conversation(title="New chat")
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _summary(row, 0)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: uuid.UUID, session: SessionDep):
    result = await session.execute(
        select(Conversation)
        .options(selectinload(Conversation.turns))
        .where(Conversation.id == conversation_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    turns = sorted(row.turns, key=lambda item: item.turn_index)
    return ConversationDetail(
        id=row.id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
        turn_count=len(turns),
        turns=[
            ConversationTurnResponse(
                id=item.id,
                turn_index=item.turn_index,
                payload=item.payload,
                created_at=item.created_at,
            )
            for item in turns
        ],
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: uuid.UUID, session: SessionDep):
    row = await session.get(Conversation, conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await session.delete(row)
    await session.commit()
