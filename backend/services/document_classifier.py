from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

DocumentKind = Literal["paper", "note"]

MAX_PAGES = 3
MAX_CHARS = 8000
PAPER_SCORE_THRESHOLD = 3
# DOI / arXiv in a bibliography must not identify the document as a paper.
HEADER_CHARS = 1500

_ABSTRACT = re.compile(r"(?im)^\s*abstract\b")
_REFERENCES = re.compile(r"(?im)^\s*(references|bibliography)\b")
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_ARXIV_ABS = re.compile(r"arxiv\.org/(?:abs|pdf)/\d{4}\.\d{4,5}", re.IGNORECASE)
_ARXIV_ID = re.compile(r"arxiv\s*:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
_INTRODUCTION = re.compile(r"(?im)^\s*(?:\d+\.?\s*)?introduction\b")
_METHODS = re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(methods?|methodology)\b")
_CONCLUSION = re.compile(r"(?im)^\s*(?:\d+\.?\s*)?conclusions?\b")
_AFFILIATION = re.compile(
    r"\b(university|institute|department|college|laboratory)\b",
    re.IGNORECASE,
)
_ET_AL = re.compile(r"\bet al\.?\b", re.IGNORECASE)
_BRACKET_CITE = re.compile(r"\[\d{1,3}(?:\s*[-–,]\s*\d{1,3})?\]")
_NOTE_TONE = re.compile(
    r"\b(TODO|meeting notes|my notes|I think)\b",
    re.IGNORECASE,
)


def classify_pdf(source: bytes | str | Path) -> DocumentKind:
    """Decide paper vs note from PDF text. Empty / unreadable PDFs are notes."""
    return classify_text(preview_text(source))


def classify_text(text: str) -> DocumentKind:
    cleaned = (text or "").strip()
    if not cleaned:
        return "note"
    return "paper" if _paper_score(cleaned) >= PAPER_SCORE_THRESHOLD else "note"


def preview_text(source: bytes | str | Path) -> str:
    try:
        if isinstance(source, bytes):
            reader = PdfReader(io.BytesIO(source))
        else:
            reader = PdfReader(str(source))
    except Exception:
        return ""

    parts: list[str] = []
    for page in reader.pages[:MAX_PAGES]:
        raw = page.extract_text() or ""
        cleaned = "\n".join(line.strip() for line in raw.splitlines())
        if cleaned.strip():
            parts.append(cleaned)
    return "\n\n".join(parts).strip()[:MAX_CHARS]


def _paper_score(text: str) -> int:
    score = 0
    if _ABSTRACT.search(text):
        score += 3
    if _REFERENCES.search(text):
        score += 3
    header = text[:HEADER_CHARS]
    if _DOI.search(header):
        score += 3
    if _has_arxiv_identity(header):
        score += 3
    if _INTRODUCTION.search(text):
        score += 1
    if _METHODS.search(text):
        score += 1
    if _CONCLUSION.search(text):
        score += 1
    if _AFFILIATION.search(text):
        score += 1
    if _ET_AL.search(text):
        score += 1
    if len(_BRACKET_CITE.findall(text)) >= 2:
        score += 1
    if _NOTE_TONE.search(text):
        score -= 2
    if len(text) < 1500 and not _ABSTRACT.search(text):
        score -= 2
    return score


def _has_arxiv_identity(header: str) -> bool:
    """True when the document itself is an arXiv preprint, not a cited one."""
    if _ARXIV_ABS.search(header):
        return True
    for match in _ARXIV_ID.finditer(header):
        prefix = header[max(0, match.start() - 40) : match.start()].lower()
        if "preprint" in prefix:
            continue
        return True
    return False
