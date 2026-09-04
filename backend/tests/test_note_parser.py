from pathlib import Path

import pytest

from services.note_parser import NoteParseError, NotePdfParser
from services.note_pipeline import voice_text_for_embedding


def build_simple_pdf(text: str) -> bytes:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 24 Tf 72 720 Td ({safe}) Tj ET\n".encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{index} 0 obj\n".encode())
        if index == 4:
            out.extend(f"<< /Length {len(stream)} >>\nstream\n".encode())
            out.extend(stream)
            out.extend(b"\nendstream\nendobj\n")
        else:
            assert obj is not None
            out.extend(obj)
            out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode()
    )
    return bytes(out)


def test_note_parser_extracts_page_text(tmp_path: Path):
    pdf_path = tmp_path / "note.pdf"
    pdf_path.write_bytes(build_simple_pdf("Hello research notes"))
    parsed = NotePdfParser().parse(pdf_path, "fallback-title")
    assert parsed.title == "fallback-title"
    assert parsed.page_count == 1
    assert "Hello research notes" in parsed.extracted_text
    assert parsed.chunks
    assert parsed.chunks[0].section == "Page 1"
    assert parsed.chunks[0].page == 1


def test_note_parser_rejects_empty_pdf(tmp_path: Path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(build_simple_pdf(""))
    with pytest.raises(NoteParseError, match="no extractable text"):
        NotePdfParser().parse(pdf_path, "empty")


def test_voice_text_for_embedding_joins_fields():
    text = voice_text_for_embedding(
        title="Hybrid retrieval",
        summary="Try BM25 plus dense search.",
        observations=["Dense search misses exact terms"],
        hypotheses=[],
        questions=["What reranker?"],
        next_steps=[],
        cleaned_transcript="I think hybrid retrieval might help.",
    )
    assert "Hybrid retrieval" in text
    assert "Observations: Dense search misses exact terms" in text
    assert "What reranker?" in text
    assert "I think hybrid retrieval might help." in text
