"""LLM digest of a parsed paper, stored for later library overviews."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import DigestStatus, Paper, PaperChunk
from schemas import PaperDigest

PROMPT_FILE = Path(__file__).resolve().parent.parent / "paper_digest_prompt.txt"
DIGEST_PROMPT = PROMPT_FILE.read_text().strip()
PROMPT_VERSION = "paper-digest-v1"
DIGEST_MAX_TOKENS = 800
DIGEST_TEMPERATURE = 0.0
DIGEST_CHAR_BUDGET = 12000
DIGEST_KEYS = ("problem", "method", "key_results", "limitations")

PREFERRED_SECTION_MARKERS = (
    "abstract",
    "introduction",
    "intro",
    "conclusion",
    "conclusions",
    "method",
    "methods",
    "approach",
    "discussion",
)

DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "problem": {"type": "string"},
        "method": {"type": "string"},
        "key_results": {"type": "string"},
        "limitations": {"type": "string"},
    },
    "required": ["summary", "problem", "method", "key_results", "limitations"],
}


class DigestChunk(Protocol):
    text: str
    section: str | None


class DigestError(RuntimeError):
    """Raised when the digest model returns nothing usable."""


def normalize_digest(value: object | None) -> dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    return {key: str(raw.get(key) or "").strip() for key in DIGEST_KEYS}


def paper_digest_from(value: object | None) -> PaperDigest:
    fields = normalize_digest(value)
    return PaperDigest(**fields)


def _section_rank(section: str | None) -> int:
    lowered = (section or "").lower()
    for index, marker in enumerate(PREFERRED_SECTION_MARKERS):
        if marker in lowered:
            return index
    return len(PREFERRED_SECTION_MARKERS)


def select_digest_source_text(
    *,
    title: str,
    authors: list[str],
    year: int | None,
    abstract: str | None,
    chunks: list[DigestChunk],
    char_budget: int = DIGEST_CHAR_BUDGET,
) -> str:
    """Title/abstract plus a capped slice that prefers intro/method/conclusion."""
    parts = [f"Title: {title.strip() or 'Untitled'}"]
    if authors:
        parts.append("Authors: " + ", ".join(authors))
    if year is not None:
        parts.append(f"Year: {year}")
    if abstract and abstract.strip():
        parts.append(f"Abstract:\n{abstract.strip()}")

    used: set[int] = set()
    remaining = max(0, char_budget - sum(len(part) for part in parts))
    ordered = sorted(
        enumerate(chunks), key=lambda item: (_section_rank(item[1].section), item[0])
    )
    selected: list[str] = []
    for index, chunk in ordered:
        text = (chunk.text or "").strip()
        if not text or index in used or remaining <= 0:
            continue
        header = chunk.section.strip() if chunk.section else f"chunk {index}"
        block = f"{header}:\n{text}"
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        if not block:
            continue
        selected.append(block)
        remaining -= len(block)
        used.add(index)
    if selected:
        parts.append("Excerpts:\n" + "\n\n".join(selected))
    return "\n".join(parts)


def _extract_payload(llm_client: object, llm_model: str, source_text: str) -> dict:
    from services.llm_chat import complete_json_object, effort_from_env

    effort = effort_from_env(
        "DIGEST_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
    )
    messages = [
        {"role": "system", "content": DIGEST_PROMPT},
        {"role": "user", "content": source_text},
    ]
    return complete_json_object(
        llm_client,
        messages=messages,
        model=llm_model,
        schema=DIGEST_SCHEMA,
        temperature=DIGEST_TEMPERATURE,
        max_tokens=DIGEST_MAX_TOKENS,
        effort=effort,
    )


def parse_digest_payload(payload: object) -> tuple[str, dict[str, str]]:
    if not isinstance(payload, dict):
        raise DigestError("digest payload is not an object")
    summary = str(payload.get("summary") or "").strip()
    digest = normalize_digest(payload)
    if not summary and not any(digest.values()):
        raise DigestError("digest payload is empty")
    return summary, digest


async def generate_and_store_digest(
    session: AsyncSession,
    paper: Paper,
    *,
    llm_client: object,
    llm_model: str,
) -> Paper:
    """Persist a digest. Failures set digest_status=failed, not processing_status."""
    paper.digest_status = DigestStatus.pending.value
    paper.updated_at = datetime.now(UTC)
    await session.commit()

    result = await session.execute(
        select(PaperChunk)
        .where(PaperChunk.paper_id == paper.id)
        .order_by(PaperChunk.chunk_index.asc())
    )
    chunks = list(result.scalars().all())
    source = select_digest_source_text(
        title=paper.title,
        authors=list(paper.authors or []),
        year=paper.year,
        abstract=paper.abstract,
        chunks=chunks,
    )
    try:
        payload = await asyncio.to_thread(
            _extract_payload, llm_client, llm_model, source
        )
        summary, digest = parse_digest_payload(payload)
    except (Exception, StopIteration):
        paper.digest_status = DigestStatus.failed.value
        paper.updated_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(paper)
        return paper

    paper.summary = summary or None
    paper.digest = digest
    paper.digest_status = DigestStatus.ready.value
    paper.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(paper)
    return paper
