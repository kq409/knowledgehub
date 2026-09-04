import re

_ABSTRACT_HEADING = re.compile(r"^abstract\b", re.IGNORECASE)
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_EMAIL = re.compile(r"\S+@\S+")
_AFFILIATION = re.compile(
    r"\b(university|institute|department|college|laboratory|ltd|inc\.?|gmbh)\b",
    re.IGNORECASE,
)


def is_abstract_heading(text: str) -> bool:
    cleaned = " ".join(text.split())
    if not cleaned or len(cleaned) > 40:
        return False
    return bool(_ABSTRACT_HEADING.match(cleaned))


def strip_abstract_prefix(text: str) -> str:
    body = _ABSTRACT_HEADING.sub("", text, count=1)
    return body.lstrip(" :.—-").strip()


def abstract_body_from_paragraph(text: str) -> str | None:
    cleaned = " ".join(text.split())
    if not _ABSTRACT_HEADING.match(cleaned):
        return None
    body = strip_abstract_prefix(cleaned)
    return body or None


def authors_from_lines(lines: list[str]) -> list[str]:
    authors: list[str] = []
    for line in lines:
        line = _EMAIL.sub("", line)
        line = " ".join(line.split())
        if not line or _AFFILIATION.search(line):
            continue
        if len(line) > 240:
            continue
        for part in re.split(r"\s*(?:,|;|&| and )\s*", line, flags=re.IGNORECASE):
            name = part.strip(" .")
            words = name.split()
            if len(words) < 2 or len(words) > 6:
                continue
            if any(ch.isdigit() for ch in name):
                continue
            if name not in authors:
                authors.append(name)
    return authors[:20]


def year_from_texts(texts: list[str]) -> int | None:
    for text in texts:
        match = _YEAR.search(text)
        if match:
            year = int(match.group(1))
            if 1950 <= year <= 2030:
                return year
    return None
