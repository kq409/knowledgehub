import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models import Note, NoteSourceType
from routers.notes import create_note, delete_note, list_notes, update_note
from schemas import (
    ExtractRequest,
    ExtractResponse,
    NoteCreate,
    NoteUpdate,
    VoiceNoteCreate,
    VoiceNoteResponse,
    VoiceNoteUpdate,
)
from schemas import (
    NoteSourceType as NoteSourceTypeSchema,
)
from services.extraction import ExtractionService, NoteExtractionError

router = APIRouter(prefix="/api/voice-notes", tags=["voice-notes"])


def get_extraction_service(request: Request) -> ExtractionService:
    service = getattr(request.app.state, "extraction", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Extraction service not ready")
    return service


SessionDep = Annotated[AsyncSession, Depends(get_session)]
ExtractionDep = Annotated[ExtractionService, Depends(get_extraction_service)]


def _as_voice_response(note: Note) -> VoiceNoteResponse:
    return VoiceNoteResponse.model_validate(note)


async def _require_voice_note(session: AsyncSession, note_id: uuid.UUID) -> Note:
    note = await session.get(Note, note_id)
    if note is None or note.source_type != NoteSourceType.voice.value:
        raise HTTPException(status_code=404, detail="Voice note not found")
    return note


@router.post("/extract", response_model=ExtractResponse)
async def extract_voice_note(
    payload: ExtractRequest,
    extraction: ExtractionDep,
):
    try:
        note = extraction.extract(
            cleaned_text=payload.cleaned_text, raw_text=payload.raw_text
        )
    except NoteExtractionError as exc:
        print(f"❌ Note extraction failed: {exc}")
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract a structured research note. "
                "Your transcription is unaffected."
            ),
        ) from exc
    except Exception as exc:
        print(f"❌ Note extraction failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail=(
                "Note extraction failed. Check the backend terminal for details. "
                "Your transcription is unaffected."
            ),
        ) from exc

    return ExtractResponse(
        **note.model_dump(),
        model=extraction.llm_model,
        prompt_version=extraction.prompt_version,
    )


@router.post("", response_model=VoiceNoteResponse, status_code=201)
async def create_voice_note(
    request: Request,
    session: SessionDep,
    payload: VoiceNoteCreate,
):
    created = await create_note(
        request, session, NoteCreate.model_validate(payload.model_dump())
    )
    note = await session.get(Note, created.id)
    if note is None:
        raise HTTPException(status_code=500, detail="Voice note was not saved")
    return _as_voice_response(note)


@router.get("", response_model=list[VoiceNoteResponse])
async def list_voice_notes(session: SessionDep):
    notes = await list_notes(session, source_type=NoteSourceTypeSchema.voice)
    return [VoiceNoteResponse.model_validate(item.model_dump()) for item in notes]


@router.get("/{note_id}", response_model=VoiceNoteResponse)
async def get_voice_note(note_id: uuid.UUID, session: SessionDep):
    note = await _require_voice_note(session, note_id)
    return _as_voice_response(note)


@router.patch("/{note_id}", response_model=VoiceNoteResponse)
async def update_voice_note(
    request: Request,
    note_id: uuid.UUID,
    payload: VoiceNoteUpdate,
    session: SessionDep,
):
    await _require_voice_note(session, note_id)
    updated = await update_note(
        request,
        note_id,
        NoteUpdate.model_validate(payload.model_dump(exclude_unset=True)),
        session,
    )
    note = await session.get(Note, updated.id)
    if note is None:
        raise HTTPException(status_code=404, detail="Voice note not found")
    return _as_voice_response(note)


@router.delete("/{note_id}", status_code=204)
async def delete_voice_note(note_id: uuid.UUID, session: SessionDep):
    await _require_voice_note(session, note_id)
    await delete_note(note_id, session)
