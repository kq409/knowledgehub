from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from services.chunking import ParsedChunk
from services.grobid_tei import parse_tei


@dataclass
class ParsedPaper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    page_count: int | None = None
    chunks: list[ParsedChunk] = field(default_factory=list)


class GrobidError(RuntimeError):
    """Raised when GROBID cannot parse a PDF."""


class PaperParser:
    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self._base_url = (
            base_url or os.getenv("GROBID_URL") or "http://grobid:8070"
        ).rstrip("/")
        self._timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("GROBID_TIMEOUT", "180"))
        )
        print(f"📄 Paper parser using GROBID at {self._base_url}", flush=True)

    def warmup(self) -> None:
        url = f"{self._base_url}/api/isalive"
        last_error: Exception | None = None
        for _attempt in range(15):
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(url)
                if response.status_code == 200:
                    print(f"✅ GROBID ready at {self._base_url}", flush=True)
                    return
                last_error = GrobidError(
                    f"GROBID isalive returned HTTP {response.status_code}"
                )
            except Exception as exc:
                last_error = exc
        print(
            f"⚠️  GROBID not reachable at {self._base_url}: {last_error}",
            flush=True,
        )

    def parse(self, pdf_path: str | Path, fallback_title: str) -> ParsedPaper:
        tei_xml = self._process_pdf(Path(pdf_path))
        parsed = parse_tei(tei_xml, fallback_title)
        return ParsedPaper(**parsed)

    def _process_pdf(self, pdf_path: Path) -> str:
        print(f"🔄 GROBID parsing {pdf_path.name}...", flush=True)
        try:
            with (
                httpx.Client(timeout=self._timeout) as client,
                pdf_path.open("rb") as handle,
            ):
                response = client.post(
                    f"{self._base_url}/api/processFulltextDocument",
                    files={"input": (pdf_path.name, handle, "application/pdf")},
                    data={
                        "consolidateHeader": "0",
                        "consolidateCitations": "0",
                        "includeRawCitations": "0",
                        "teiCoordinates": "head,p,div,title,figDesc,abstract",
                    },
                )
        except httpx.HTTPError as exc:
            raise GrobidError(f"GROBID request failed: {exc}") from exc

        if response.status_code >= 400:
            detail = (response.text or "").strip()[:500]
            raise GrobidError(
                f"GROBID returned HTTP {response.status_code}"
                + (f": {detail}" if detail else "")
            )
        if not response.text.strip():
            raise GrobidError("GROBID returned an empty TEI document")
        return response.text
