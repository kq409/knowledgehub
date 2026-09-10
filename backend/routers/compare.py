import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models import PaperComparison
from schemas import CompareRequest, CompareResponse, CompareSummary
from services.compare import (
    CompareError,
    CompareService,
    CompareValidationError,
    response_from_row,
)
from services.identity import IdentityDep

router = APIRouter(prefix="/api/compare", tags=["compare"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_compare_service(request: Request) -> CompareService:
    service = getattr(request.app.state, "compare", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Compare service not ready")
    return service


CompareDep = Annotated[CompareService, Depends(get_compare_service)]


def _summary_from_row(row: PaperComparison) -> CompareSummary:
    papers = (row.result or {}).get("papers") or []
    titles = [str(item.get("title") or "") for item in papers if isinstance(item, dict)]
    paper_ids = [uuid.UUID(str(item)) for item in (row.paper_ids or [])]
    return CompareSummary(
        id=row.id,
        paper_ids=paper_ids,
        paper_titles=titles,
        created_at=row.created_at,
    )


@router.post("", response_model=CompareResponse)
@router.post("/", response_model=CompareResponse, include_in_schema=False)
async def create_comparison(
    payload: CompareRequest,
    session: SessionDep,
    compare_service: CompareDep,
    identity: IdentityDep,
):
    try:
        return await compare_service.compare(
            session, payload, space_ids=identity.space_ids
        )
    except CompareValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CompareError as exc:
        print(f"❌ Compare generation failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Could not generate a comparison. Check the backend terminal for details.",
        ) from exc
    except Exception as exc:
        print(f"❌ Compare failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Compare failed. Check the backend terminal for details.",
        ) from exc


@router.get("", response_model=list[CompareSummary])
@router.get("/", response_model=list[CompareSummary], include_in_schema=False)
async def list_comparisons(session: SessionDep):
    result = await session.execute(
        select(PaperComparison).order_by(PaperComparison.created_at.desc())
    )
    return [_summary_from_row(row) for row in result.scalars().all()]


@router.get("/{comparison_id}", response_model=CompareResponse)
async def get_comparison(comparison_id: uuid.UUID, session: SessionDep):
    row = await session.get(PaperComparison, comparison_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Comparison not found")
    try:
        return response_from_row(row)
    except Exception as exc:
        print(f"❌ Compare load failed: {exc}")
        raise HTTPException(
            status_code=500, detail="Saved comparison could not be loaded."
        ) from exc


@router.delete("/{comparison_id}", status_code=204)
async def delete_comparison(comparison_id: uuid.UUID, session: SessionDep):
    row = await session.get(PaperComparison, comparison_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Comparison not found")
    await session.delete(row)
    await session.commit()
