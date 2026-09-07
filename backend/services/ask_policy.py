from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from schemas import LibraryCoverage, QueryKind
from services.retrieval import LibraryPaper, RetrievalHit, filter_by_min_similarity

FIELD_WIDE_RECENCY_WINDOW = 3

_FIELD_WIDE_RE = re.compile(
    r"""
    \bsota\b
    | state[\s-]*of[\s-]*the[\s-]*art
    | \blatest\b
    | \bmost\s+recent\b
    | so\s+far\s+in\s+(?:19|20)\d{2}
    | \bas\s+of\s+(?:19|20)\d{2}
    | \bbest\s+(?:algorithm|method|approach|model)
    | \bcurrently\s+(?:the\s+)?best\b
    | 最先进
    | 目前最好
    | 目前最(?:先进|好|前沿)
    | 最新(?:进展|算法|方法)
    | 最前沿
    """,
    re.IGNORECASE | re.VERBOSE,
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class EvidenceQuality(str, Enum):
    unsupported = "UNSUPPORTED"
    low = "LOW"
    ok = "OK"


@dataclass(frozen=True)
class AskDecision:
    query_kind: QueryKind
    quality: EvidenceQuality
    skip_llm: bool
    suggest_external_search: bool
    insufficient_evidence: bool
    coverage: LibraryCoverage
    max_similarity: float | None
    generation_hits: tuple[RetrievalHit, ...]
    citation_hits: tuple[RetrievalHit, ...]


def classify_query(question: str) -> QueryKind:
    if _FIELD_WIDE_RE.search(question or ""):
        return QueryKind.field_wide
    return QueryKind.library


def uses_chinese(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def named_years(question: str) -> list[int]:
    years: list[int] = []
    for match in _YEAR_RE.finditer(question or ""):
        year = int(match.group(1))
        if 1950 <= year <= 2100 and year not in years:
            years.append(year)
    return years


def current_year() -> int:
    return datetime.now(UTC).year


def question_names_recent_year(question: str, now_year: int | None = None) -> bool:
    cutoff = now_year if now_year is not None else current_year()
    window_start = cutoff - FIELD_WIDE_RECENCY_WINDOW
    return any(year >= window_start for year in named_years(question))


def unique_retrieved_paper_ids(hits: list[RetrievalHit]) -> set[uuid.UUID]:
    return {hit.source_id for hit in hits if hit.source_type == "paper"}


def library_years(papers: list[LibraryPaper]) -> list[int]:
    return sorted({paper.year for paper in papers if paper.year is not None})


def coverage_from(
    papers: list[LibraryPaper], hits: list[RetrievalHit]
) -> LibraryCoverage:
    return LibraryCoverage(
        paper_count=len(papers),
        unique_retrieved_papers=len(unique_retrieved_paper_ids(hits)),
        years=library_years(papers),
    )


def inventory_line(papers: list[LibraryPaper], hits: list[RetrievalHit]) -> str:
    labels: list[str] = []
    for paper in papers[:8]:
        if paper.year is not None:
            labels.append(f"{paper.title} ({paper.year})")
        else:
            labels.append(paper.title)
    label_part = ", ".join(labels) if labels else "none"
    years = library_years(papers)
    year_part = ", ".join(str(year) for year in years) if years else "unknown"
    unique_papers = len(unique_retrieved_paper_ids(hits))
    return (
        f"Library: {len(papers)} paper(s) ({label_part}). "
        f"Retrieved {len(hits)} chunk(s) from {unique_papers} paper(s). "
        f"Years: {year_part}."
    )


def _papers_cover_recent_years(papers: list[LibraryPaper], now_year: int) -> bool:
    years = library_years(papers)
    if not years:
        return False
    return max(years) >= now_year - FIELD_WIDE_RECENCY_WINDOW


def field_wide_unsupported(
    papers: list[LibraryPaper],
    hits: list[RetrievalHit],
    filtered_hits: list[RetrievalHit],
    question: str,
    now_year: int | None = None,
) -> bool:
    year = now_year if now_year is not None else current_year()
    if len(papers) <= 1:
        return True
    if len(unique_retrieved_paper_ids(hits)) <= 1:
        return True
    if not filtered_hits:
        return True
    return question_names_recent_year(
        question, year
    ) and not _papers_cover_recent_years(papers, year)


def decide_ask(
    question: str,
    hits: list[RetrievalHit],
    papers: list[LibraryPaper],
    min_similarity: float,
    now_year: int | None = None,
) -> AskDecision:
    query_kind = classify_query(question)
    filtered = filter_by_min_similarity(hits, min_similarity)
    max_similarity = max((hit.similarity for hit in hits), default=None)
    coverage = coverage_from(papers, hits)

    if query_kind == QueryKind.field_wide and field_wide_unsupported(
        papers, hits, filtered, question, now_year=now_year
    ):
        citation_hits = tuple(filtered or hits)
        return AskDecision(
            query_kind=query_kind,
            quality=EvidenceQuality.unsupported,
            skip_llm=True,
            suggest_external_search=True,
            insufficient_evidence=True,
            coverage=coverage,
            max_similarity=max_similarity,
            generation_hits=(),
            citation_hits=citation_hits,
        )

    if not hits:
        return AskDecision(
            query_kind=query_kind,
            quality=EvidenceQuality.unsupported,
            skip_llm=True,
            suggest_external_search=False,
            insufficient_evidence=True,
            coverage=coverage,
            max_similarity=None,
            generation_hits=(),
            citation_hits=(),
        )

    if not filtered:
        ranked = tuple(hits)
        return AskDecision(
            query_kind=query_kind,
            quality=EvidenceQuality.low,
            skip_llm=False,
            suggest_external_search=False,
            insufficient_evidence=True,
            coverage=coverage,
            max_similarity=max_similarity,
            generation_hits=ranked,
            citation_hits=ranked,
        )

    kept = tuple(filtered)
    return AskDecision(
        query_kind=query_kind,
        quality=EvidenceQuality.ok,
        skip_llm=False,
        suggest_external_search=False,
        insufficient_evidence=False,
        coverage=coverage,
        max_similarity=max_similarity,
        generation_hits=kept,
        citation_hits=kept,
    )


def _closest_match_lines(hits: tuple[RetrievalHit, ...]) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for hit in hits:
        key = f"{hit.source_type}:{hit.source_id}"
        if key in seen:
            continue
        seen.add(key)
        year = f" ({hit.year})" if hit.year is not None else ""
        lines.append(f"- {hit.title}{year}")
        if len(lines) >= 5:
            break
    return lines


def abstain_answer(question: str, decision: AskDecision) -> str:
    chinese = uses_chinese(question)
    years = decision.coverage.years
    year_text = (
        ", ".join(str(year) for year in years)
        if years
        else ("未知" if chinese else "unknown")
    )
    paper_count = decision.coverage.paper_count
    closest = _closest_match_lines(decision.citation_hits)

    if decision.query_kind == QueryKind.field_wide:
        if chinese:
            parts = [
                f"仅根据你的文献库（{paper_count} 篇论文，年份 {year_text}）。",
                "",
                "这个问题需要领域级或最新进展（例如 SOTA）的判断。"
                "当前库的覆盖不足以支持该结论。",
            ]
            if closest:
                parts.extend(["", "库中最接近的内容：", *closest])
            parts.extend(["", "需要领域级结论时，可在配置库外搜索后继续检索网络。"])
            return "\n".join(parts)
        parts = [
            f"Based only on your library ({paper_count} paper(s), years {year_text}).",
            "",
            "This question asks for field-wide or latest progress (for example SOTA). "
            "The library does not have enough coverage to support that claim.",
        ]
        if closest:
            parts.extend(["", "Closest matches in your library:", *closest])
        parts.extend(
            [
                "",
                "For a field-wide answer, search outside the library if web search is configured.",
            ]
        )
        return "\n".join(parts)

    if chinese:
        return "文献库中没有找到支持该问题的证据，因此无法作答。"
    return (
        "No supporting evidence was found in your library. "
        "The library does not support an answer to this question."
    )


def quality_instruction(quality: EvidenceQuality) -> str:
    if quality is EvidenceQuality.unsupported:
        return (
            "UNSUPPORTED — do not answer the claim; say the library cannot support it."
        )
    if quality is EvidenceQuality.low:
        return (
            "LOW — answer only what the library supports, clearly state the limit, "
            "and do not invent."
        )
    return "OK — answer from the evidence, still scoped to this library."
