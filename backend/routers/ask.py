from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import APIError, NotFoundError
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


def _looks_like_embedding_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "nomic-embed",
            "embedding",
            "/v1/embeddings",
            "embed",
        )
    )


def ask_http_error(exc: BaseException) -> HTTPException:
    """Map Ask failures to a short, actionable 502 without leaking secrets."""
    if isinstance(exc, AskError):
        return HTTPException(
            status_code=502,
            detail=(
                "Could not generate an answer (empty model content). "
                "For thinking models try ASK_REASONING_EFFORT=none "
                "(or LLM_REASONING_EFFORT=none) and raise ASK_MAX_TOKENS."
            ),
        )

    # NotFound on Ask is usually nomic-embed-text hitting a chat-only cloud host.
    if isinstance(exc, NotFoundError) or _looks_like_embedding_error(exc):
        return HTTPException(
            status_code=502,
            detail=(
                "Embedding request failed (model or host not found). "
                "Check EMBEDDING_BASE_URL / EMBEDDING_MODEL — keep "
                "embeddings on Ollama when chat uses a cloud host "
                "without embedding models."
            ),
        )

    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        status_bit = f" (HTTP {status})" if status else ""
        return HTTPException(
            status_code=502,
            detail=f"LLM API error{status_bit}. Check LLM_BASE_URL / LLM_MODEL.",
        )

    return HTTPException(
        status_code=502,
        detail="Ask failed. Check the backend terminal for details.",
    )


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
        raise ask_http_error(exc) from exc
    except (NotFoundError, APIError) as exc:
        print(f"❌ Ask API failed: {exc}")
        raise ask_http_error(exc) from exc
    except Exception as exc:
        print(f"❌ Ask failed: {exc}")
        raise ask_http_error(exc) from exc
