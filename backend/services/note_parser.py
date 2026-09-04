from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from services.chunking import ParsedChunk, split_with_overlap


class NoteParseError(RuntimeError):
    """Raised when a note PDF has no extractable text."""


@dataclass
class ParsedNotePdf:
    title: str
    extracted_text: str
    page_count: int
    chunks: list[ParsedChunk] = field(default_factory=list)


class NotePdfParser:
    def parse(self, pdf_path: str | Path, fallback_title: str) -> ParsedNotePdf:
        path = Path(pdf_path)
        try:
            reader = PdfReader(str(path))
        except PdfReadError as exc:
            raise NoteParseError(f"Could not read PDF: {exc}") from exc
        except Exception as exc:
            raise NoteParseError(f"Could not open PDF: {exc}") from exc

        chunks: list[ParsedChunk] = []
        page_texts: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = " ".join(raw.split())
            if not text:
                continue
            page_texts.append(text)
            for part in split_with_overlap(text):
                chunks.append(
                    ParsedChunk(text=part, page=index, section=f"Page {index}")
                )

        extracted = "\n\n".join(page_texts).strip()
        if not extracted:
            raise NoteParseError("PDF has no extractable text")

        return ParsedNotePdf(
            title=(fallback_title or "Untitled note")[:500],
            extracted_text=extracted,
            page_count=len(reader.pages),
            chunks=chunks,
        )
