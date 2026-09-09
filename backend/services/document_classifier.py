from __future__ import annotations

import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

DocumentKind = Literal["paper", "note"]
AttachmentKind = Literal["paper", "note", "document"]

MAX_PAGES = 3
MAX_CHARS = 8000
PAPER_SCORE_THRESHOLD = 3
# DOI / arXiv in a bibliography must not identify the document as a paper.
HEADER_CHARS = 1500
SCORE_CLOSE_DELTA = 1
PREVIEW_HEAD_CHARS = 4000

DOCUMENT_SUFFIXES = {".csv", ".md", ".txt", ".docx"}
PDF_SUFFIXES = {".pdf"}

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
_PAPER_PROMPT = re.compile(
    r"(as a paper|file (?:this|it) as a paper|当成(?:一篇)?论文|这是论文|当成 paper)",
    re.IGNORECASE,
)
_NOTE_PROMPT = re.compile(
    r"(my notes|handwritten|meeting notes|as (?:a )?note|"
    r"当成笔记|这是我的笔记|手写笔记)",
    re.IGNORECASE,
)
_DOCUMENT_PROMPT = re.compile(
    r"(spreadsheet|as a document|generic document|markdown file|"
    r"实验表格|资料文件|当成文档|这是表格|这是 csv)",
    re.IGNORECASE,
)


def classify_pdf(source: bytes | str | Path) -> DocumentKind:
    """Decide paper vs note from PDF text. Empty / unreadable PDFs are notes."""
    return classify_text(preview_text(source))


def classify_text(text: str) -> DocumentKind:
    cleaned = (text or "").strip()
    if not cleaned:
        return "note"
    return "paper" if paper_score(cleaned) >= PAPER_SCORE_THRESHOLD else "note"


def paper_score(text: str) -> int:
    return _paper_score(text)


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


def attachment_preview(
    filename: str,
    content: bytes,
    content_type: str | None = None,
) -> str:
    suffix = Path(filename or "").suffix.lower()
    ctype = (content_type or "").lower()
    if suffix == ".pdf" or "pdf" in ctype:
        return preview_text(content)
    if suffix == ".docx" or "wordprocessingml" in ctype:
        return _docx_preview(content)
    try:
        return content.decode("utf-8", errors="replace").strip()[:PREVIEW_HEAD_CHARS]
    except Exception:
        return ""


def prompt_kind(prompt: str) -> AttachmentKind | None:
    text = (prompt or "").strip()
    if not text:
        return None
    hits: list[AttachmentKind] = []
    if _PAPER_PROMPT.search(text):
        hits.append("paper")
    if _NOTE_PROMPT.search(text):
        hits.append("note")
    if _DOCUMENT_PROMPT.search(text):
        hits.append("document")
    if len(hits) == 1:
        return hits[0]
    return None


def classify_attachment(
    *,
    filename: str,
    content: bytes,
    content_type: str | None = None,
    prompt: str = "",
    llm_classify: Callable[[str], AttachmentKind] | None = None,
) -> AttachmentKind:
    """File this attachment as paper, handwritten note, or generic document.

    Extension sets the default for non-PDFs. A clear prompt can override.
    PDF uses the existing heuristic; a conflicting or borderline prompt+score
    asks `llm_classify` (prompt + preview) and falls back if that fails.
    """
    suffix = Path(filename or "").suffix.lower()
    ctype = (content_type or "").lower()
    override = prompt_kind(prompt)
    preview = attachment_preview(filename, content, content_type)

    if suffix in DOCUMENT_SUFFIXES or _looks_like_office_text(suffix, ctype):
        baseline: AttachmentKind = "document"
        if override is None or override == baseline:
            return baseline
        return _maybe_llm(
            llm_classify,
            prompt=prompt,
            filename=filename,
            preview=preview,
            fallback=override,
        )

    if suffix in PDF_SUFFIXES or "pdf" in ctype:
        heuristic = classify_text(preview)
        score = paper_score(preview) if preview.strip() else 0
        close = abs(score - PAPER_SCORE_THRESHOLD) <= SCORE_CLOSE_DELTA
        if override is None:
            if close and prompt.strip() and llm_classify is not None:
                return _maybe_llm(
                    llm_classify,
                    prompt=prompt,
                    filename=filename,
                    preview=preview,
                    fallback=heuristic,
                )
            return heuristic
        if override == heuristic and not close:
            return heuristic
        fallback: AttachmentKind = override
        return _maybe_llm(
            llm_classify,
            prompt=prompt,
            filename=filename,
            preview=preview,
            fallback=fallback,
        )

    if override is not None:
        return override
    return "document"


def _looks_like_office_text(suffix: str, ctype: str) -> bool:
    if suffix in DOCUMENT_SUFFIXES:
        return True
    if "csv" in ctype or "markdown" in ctype or ctype.startswith("text/"):
        return True
    return "wordprocessingml" in ctype


def _maybe_llm(
    llm_classify: Callable[[str], AttachmentKind] | None,
    *,
    prompt: str,
    filename: str,
    preview: str,
    fallback: AttachmentKind,
) -> AttachmentKind:
    if llm_classify is None:
        return fallback
    payload = (
        f"filename: {filename}\n"
        f"researcher prompt: {prompt.strip() or '(none)'}\n"
        f"preview:\n{preview[:3000]}"
    )
    try:
        kind = llm_classify(payload)
    except Exception:
        return fallback
    if kind in ("paper", "note", "document"):
        return kind
    return fallback


def _docx_preview(content: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        return ""
    try:
        document = Document(io.BytesIO(content))
    except Exception:
        return ""
    parts = [para.text.strip() for para in document.paragraphs if para.text.strip()]
    return "\n".join(parts)[:PREVIEW_HEAD_CHARS]


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
