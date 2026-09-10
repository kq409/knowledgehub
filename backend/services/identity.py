"""Stub-user identity for the Phase 1 ACL demo.

This is not SSO. `X-User-Id: alice|bob` selects a membership list. Missing
header defaults to alice so existing tests and the eval harness keep working.
Unknown users see an empty library rather than falling open.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models import DEFAULT_SPACE_ID, DEFAULT_SPACE_SLUG, Space, SpaceMembership

USER_HEADER = "X-User-Id"
DEFAULT_USER_ID = "alice"
STUB_USER_IDS = frozenset({"alice", "bob"})
HIDDEN_RECORD = "No record with that id is visible in your spaces."
HIDDEN_PAPER = "No paper with that id is visible in your spaces."
HIDDEN_NOTE = "No note with that id is visible in your spaces."
HIDDEN_DOCUMENT = "No document with that id is visible in your spaces."


@dataclass(frozen=True)
class Identity:
    user_id: str
    space_ids: frozenset[uuid.UUID]
    primary_space_id: uuid.UUID

    def can_see(self, space_id: uuid.UUID | None) -> bool:
        if space_id is None:
            return False
        return space_id in self.space_ids


def normalize_user_id(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    return value or DEFAULT_USER_ID


def bound_space_ids(space_ids: frozenset[uuid.UUID] | None) -> frozenset[uuid.UUID]:
    """Search never runs unrestricted. None means the default space only."""
    if space_ids is None:
        return frozenset({DEFAULT_SPACE_ID})
    return space_ids


def space_clause(column, space_ids: frozenset[uuid.UUID]):
    if not space_ids:
        return column.in_(())
    return column.in_(tuple(space_ids))


def default_identity() -> Identity:
    return Identity(
        user_id=DEFAULT_USER_ID,
        space_ids=frozenset({DEFAULT_SPACE_ID}),
        primary_space_id=DEFAULT_SPACE_ID,
    )


def eval_zxq_identity() -> Identity:
    """ZXQ eval suites stay inside the default space even if alice later joins others."""
    return Identity(
        user_id=DEFAULT_USER_ID,
        space_ids=frozenset({DEFAULT_SPACE_ID}),
        primary_space_id=DEFAULT_SPACE_ID,
    )


async def load_identity(session: AsyncSession, user_id: str) -> Identity:
    result = await session.execute(
        select(SpaceMembership.space_id).where(SpaceMembership.user_id == user_id)
    )
    space_ids = frozenset(row[0] for row in result.all())
    primary = (
        DEFAULT_SPACE_ID
        if DEFAULT_SPACE_ID in space_ids
        else next(iter(space_ids), DEFAULT_SPACE_ID)
    )
    return Identity(user_id=user_id, space_ids=space_ids, primary_space_id=primary)


async def identity_from_request(request: Request, session: AsyncSession) -> Identity:
    return await load_identity(
        session, normalize_user_id(request.headers.get(USER_HEADER))
    )


async def get_identity(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Identity:
    return await identity_from_request(request, session)


IdentityDep = Annotated[Identity, Depends(get_identity)]


def require_visible(entity, identity: Identity, *, kind: str = "Record") -> None:
    space_id = getattr(entity, "space_id", None) if entity is not None else None
    if entity is None or not identity.can_see(space_id):
        raise HTTPException(status_code=404, detail=f"{kind} not found")


async def ensure_default_space(session: AsyncSession) -> Space:
    space = await session.get(Space, DEFAULT_SPACE_ID)
    if space is not None:
        return space
    space = Space(
        id=DEFAULT_SPACE_ID,
        slug=DEFAULT_SPACE_SLUG,
        name="Default library",
    )
    session.add(space)
    for user_id in sorted(STUB_USER_IDS):
        session.add(SpaceMembership(space_id=DEFAULT_SPACE_ID, user_id=user_id))
    await session.flush()
    return space
