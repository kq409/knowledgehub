from __future__ import annotations

from xml.etree import ElementTree

from services.chunking import ParsedChunk, split_with_overlap
from services.paper_metadata import year_from_texts

TEI_NS = "http://www.tei-c.org/ns/1.0"


def q(tag: str) -> str:
    return f"{{{TEI_NS}}}{tag}"


def local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1]


def element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def page_from_coords(element: ElementTree.Element | None) -> int | None:
    if element is None:
        return None
    coords = element.get("coords")
    if not coords:
        return None
    first = coords.split(";")[0].split(",")[0].strip()
    if first.isdigit():
        return int(first)
    return None


def max_page(root: ElementTree.Element) -> int | None:
    pages = [
        page
        for element in root.iter()
        if (page := page_from_coords(element)) is not None
    ]
    return max(pages) if pages else None


def _author_name(author: ElementTree.Element) -> str | None:
    pers = author.find(q("persName"))
    if pers is None:
        return None
    forenames = [
        element_text(node) for node in pers.findall(q("forename")) if element_text(node)
    ]
    surname = element_text(pers.find(q("surname")))
    parts = [*forenames, surname] if surname else forenames
    name = " ".join(parts).strip()
    return name or None


def _extract_title(root: ElementTree.Element, fallback_title: str) -> str:
    analytic = root.find(f".//{q('analytic')}")
    if analytic is not None:
        title = element_text(analytic.find(q("title")))
        if title:
            return title[:500]
    header_title = root.find(f".//{q('titleStmt')}/{q('title')}")
    title = element_text(header_title)
    if title:
        return title[:500]
    return fallback_title[:500]


def _extract_authors(root: ElementTree.Element) -> list[str]:
    authors: list[str] = []
    analytic = root.find(f".//{q('analytic')}")
    search_root = analytic if analytic is not None else root
    for author in search_root.findall(f".//{q('author')}"):
        name = _author_name(author)
        if name and name not in authors:
            authors.append(name)
        if len(authors) >= 20:
            break
    return authors


def _extract_year(root: ElementTree.Element) -> int | None:
    texts: list[str] = []
    for date in root.findall(f".//{q('date')}"):
        when = date.get("when") or ""
        texts.append(when)
        texts.append(element_text(date))
    return year_from_texts(texts)


def _extract_abstract(root: ElementTree.Element) -> str | None:
    abstract = root.find(f".//{q('profileDesc')}/{q('abstract')}")
    text = element_text(abstract)
    return text or None


def _emit_section(
    section: str, page: int | None, paragraphs: list[str]
) -> list[ParsedChunk]:
    text = " ".join(paragraphs).strip()
    if not text:
        return []
    return [
        ParsedChunk(text=part, page=page, section=section[:512])
        for part in split_with_overlap(text)
    ]


def _walk_div(div: ElementTree.Element, parent_section: str) -> list[ParsedChunk]:
    head = next(
        (child for child in list(div) if local_tag(child.tag) == "head"),
        None,
    )
    section = element_text(head) or parent_section
    section_page = page_from_coords(head) or page_from_coords(div)
    page = section_page
    chunks: list[ParsedChunk] = []
    paragraphs: list[str] = []

    for child in list(div):
        tag = local_tag(child.tag)
        if tag == "head":
            continue
        if tag == "div":
            chunks.extend(_emit_section(section, section_page or page, paragraphs))
            paragraphs = []
            chunks.extend(_walk_div(child, section))
            continue
        if tag in {"p", "formula", "figDesc", "note"}:
            text = element_text(child)
            if text:
                paragraphs.append(text)
                page = page_from_coords(child) or page
    chunks.extend(_emit_section(section, section_page or page, paragraphs))
    return chunks


def _body_chunks(root: ElementTree.Element) -> list[ParsedChunk]:
    body = root.find(f".//{q('text')}/{q('body')}")
    if body is None:
        body = root.find(f".//{q('body')}")
    if body is None:
        return []

    chunks: list[ParsedChunk] = []
    paragraphs: list[str] = []
    page: int | None = None
    for child in list(body):
        tag = local_tag(child.tag)
        if tag == "div":
            chunks.extend(_emit_section("Document", page, paragraphs))
            paragraphs = []
            chunks.extend(_walk_div(child, "Document"))
            continue
        if tag in {"p", "formula", "figDesc", "note"}:
            text = element_text(child)
            if text:
                paragraphs.append(text)
                page = page_from_coords(child) or page
    chunks.extend(_emit_section("Document", page, paragraphs))
    return chunks


def parse_tei(tei_xml: str, fallback_title: str) -> dict:
    root = ElementTree.fromstring(tei_xml)
    title = _extract_title(root, fallback_title)
    abstract = _extract_abstract(root)
    chunks = _body_chunks(root)
    if abstract:
        abstract_chunks = [
            ParsedChunk(text=part, page=1, section="Abstract")
            for part in split_with_overlap(abstract)
        ]
        chunks = abstract_chunks + chunks
    if not chunks:
        leftover = element_text(root)
        chunks = [
            ParsedChunk(text=part, page=None, section="Document")
            for part in split_with_overlap(leftover)
        ]
    return {
        "title": title,
        "authors": _extract_authors(root),
        "year": _extract_year(root),
        "abstract": abstract,
        "page_count": max_page(root),
        "chunks": chunks,
    }
