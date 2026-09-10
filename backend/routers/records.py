from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models import LibraryDocument, Note, Paper
from routers.documents import schedule_document_processing
from routers.notes import schedule_note_processing
from routers.papers import schedule_paper_processing
from schemas import (
    CatalogSearchRequest,
    LibraryRecordResponse,
    RecordPageResponse,
    SpaceResponse,
)
from services.identity import HIDDEN_RECORD, Identity, IdentityDep, require_visible
from services.library_ingest import (
    append_revision,
    validate_library_file,
    validate_pdf_bytes,
)
from services.library_records import (
    get_record,
    list_records,
    list_visible_spaces,
    parse_content_type,
    search_records,
)
from services.storage import ImmutableObjectError

router = APIRouter(tags=["records"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _page(result) -> RecordPageResponse:
    return RecordPageResponse(
        items=[
            LibraryRecordResponse.model_validate(item, from_attributes=True)
            for item in result.items
        ],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )


def _spaces_for(identity: Identity, space_id: uuid.UUID | None) -> frozenset[uuid.UUID]:
    if space_id is None:
        return identity.space_ids
    if space_id not in identity.space_ids:
        raise HTTPException(status_code=404, detail="Space not found")
    return frozenset({space_id})


@router.get("/api/spaces", response_model=list[SpaceResponse])
async def get_spaces(session: SessionDep, identity: IdentityDep):
    rows = await list_visible_spaces(session, identity)
    return [SpaceResponse(id=row.id, slug=row.slug, name=row.name) for row in rows]


@router.get("/api/records", response_model=RecordPageResponse)
async def get_records(
    session: SessionDep,
    identity: IdentityDep,
    space_id: uuid.UUID | None = None,
    content_type: str | None = None,
    status: str | None = None,
    accession_status: str | None = None,
    limit: int | None = Query(default=None, ge=1, le=100),
    offset: int | None = Query(default=None, ge=0),
):
    try:
        parsed_type = parse_content_type(content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    page = await list_records(
        session,
        space_ids=_spaces_for(identity, space_id),
        content_type=parsed_type,
        status=status,
        accession_status=accession_status,
        limit=limit,
        offset=offset,
    )
    return _page(page)


@router.get("/api/records/{record_id}", response_model=LibraryRecordResponse)
async def get_one_record(
    record_id: uuid.UUID,
    session: SessionDep,
    identity: IdentityDep,
):
    record = await get_record(session, record_id, space_ids=identity.space_ids)
    if record is None:
        raise HTTPException(status_code=404, detail=HIDDEN_RECORD)
    return LibraryRecordResponse.model_validate(record, from_attributes=True)


@router.post(
    "/api/records/{record_id}/revisions",
    response_model=LibraryRecordResponse,
    status_code=202,
)
async def add_record_revision(
    record_id: uuid.UUID,
    request: Request,
    session: SessionDep,
    identity: IdentityDep,
    file: Annotated[UploadFile, File()],
):
    """Append a new original. Overwriting the stored bytes is refused."""
    paper = await session.get(Paper, record_id)
    note = None if paper is not None else await session.get(Note, record_id)
    document = (
        None
        if paper is not None or note is not None
        else await session.get(LibraryDocument, record_id)
    )
    entity = paper or note or document
    require_visible(entity, identity)
    content = await file.read()
    try:
        if paper is not None or note is not None:
            filename = validate_pdf_bytes(
                file.filename, file.content_type, content, fallback_name="revision.pdf"
            )
        else:
            filename = validate_library_file(
                file.filename,
                file.content_type,
                content,
                fallback_name=Path(file.filename or "revision.bin").name,
            )
        await append_revision(session, entity, content, filename)
    except ImmutableObjectError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if paper is not None:
        schedule_paper_processing(request.app, paper.id)
    elif note is not None:
        schedule_note_processing(request.app, note.id)
    else:
        if document is None:
            raise HTTPException(status_code=404, detail=HIDDEN_RECORD)
        schedule_document_processing(request.app, document.id)
    record = await get_record(session, record_id, space_ids=identity.space_ids)
    if record is None:
        raise HTTPException(status_code=404, detail=HIDDEN_RECORD)
    return LibraryRecordResponse.model_validate(record, from_attributes=True)


@router.post("/api/search", response_model=RecordPageResponse)
async def catalog_search(
    payload: CatalogSearchRequest,
    session: SessionDep,
    identity: IdentityDep,
):
    try:
        parsed_type = parse_content_type(payload.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    page = await search_records(
        session,
        payload.query,
        space_ids=_spaces_for(identity, payload.space_id),
        content_type=parsed_type,
        record_ids=payload.record_ids,
        limit=payload.limit,
        offset=payload.offset,
    )
    return _page(page)
