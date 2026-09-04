from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from schemas import AskRequest, AskResponse
from services.ask import AskError, AskService

router = APIRouter(prefix="/api/ask", tags=["ask"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_ask_service(request: Request) -> AskService:
    service = getattr(request.app.state, "ask", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Ask service not ready")
    return service


AskDep = Annotated[AskService, Depends(get_ask_service)]


@router.post("", response_model=AskResponse)
@router.post("/", response_model=AskResponse, include_in_schema=False)
async def ask_research(
    payload: AskRequest,
    session: SessionDep,
    ask_service: AskDep,
):
    if (
        not payload.include_papers
        and not payload.include_voice_notes
        and not payload.include_handwritten_notes
    ):
        raise HTTPException(
            status_code=400,
            detail="Select at least one source: papers, voice notes, or handwritten notes.",
        )

    try:
        return await ask_service.ask(session, payload)
    except AskError as exc:
        print(f"❌ Ask generation failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Could not generate an answer. Check the backend terminal for details.",
        ) from exc
    except Exception as exc:
        print(f"❌ Ask failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Ask failed. Check the backend terminal for details.",
        ) from exc
