from dataclasses import dataclass


@dataclass
class ParsedChunk:
    text: str
    page: int | None
    section: str | None


def split_with_overlap(
    text: str, *, max_chars: int = 1800, overlap: int = 200
) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    parts: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            window = cleaned[start:end]
            split_at = window.rfind(". ")
            if split_at >= max_chars // 2:
                end = start + split_at + 1
        parts.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return [part for part in parts if part]


def extract_page_from_chunk(chunk) -> int | None:
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None
    items = getattr(meta, "doc_items", None) or []
    for item in items:
        prov = getattr(item, "prov", None) or []
        if not prov:
            continue
        page = getattr(prov[0], "page_no", None)
        if page is not None:
            return int(page)
    return None


def extract_section_from_chunk(chunk) -> str | None:
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None
    headings = getattr(meta, "headings", None) or []
    headings = [heading.strip() for heading in headings if heading and heading.strip()]
    if not headings:
        return None
    return " / ".join(headings)[:512]
