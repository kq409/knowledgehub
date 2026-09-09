"""On-disk skills the agent loads into context on demand.

Skills live as markdown files under backend/skills/ with YAML frontmatter
(name, description) and a body. Only the one-line descriptions go into the
system prompt, as a catalog; the bodies stay on disk until `load_skill` pulls
one in. That split is the whole point: five full skills would cost more
context than the conversation, while five descriptions cost five lines and are
enough for the model to know what exists.

The catalog is read once at startup and cached. It used to be re-read from disk
on every `list_skills` and every `load_skill`, which meant a directory scan and
five file reads inside a request that was already waiting on a model. Call
`reload_skills()` after editing a skill file.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path

    def full_text(self) -> str:
        return f"# Skill: {self.name}\n\n{self.description}\n\n{self.body}".strip()


class SkillError(ValueError):
    """Raised when a skill cannot be loaded."""


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text.strip()
    rest = text[3:].lstrip("\n")
    end = rest.find("\n---")
    if end < 0:
        return {}, text.strip()
    header = rest[:end]
    body = rest[end + 4 :].strip()
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip("\"'")
    return meta, body


def _scan(root: Path) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    if not root.is_dir():
        return skills
    for path in sorted(root.glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        name = (meta.get("name") or path.stem).strip()
        description = (meta.get("description") or "").strip()
        if not name:
            continue
        skills[name] = Skill(name=name, description=description, body=body, path=path)
    return skills


_cache: dict[str, Skill] | None = None
_lock = threading.Lock()


def load_all_skills(directory: Path | None = None) -> dict[str, Skill]:
    """The skill catalog. Cached unless an explicit directory is given.

    Tests pass a directory to read a fixture, and those reads deliberately
    bypass the cache so one test cannot poison another.
    """
    if directory is not None:
        return _scan(directory)

    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _scan(SKILLS_DIR)
    return _cache


def reload_skills() -> dict[str, Skill]:
    """Re-read the skills directory. For startup and for editing a skill."""
    global _cache
    with _lock:
        _cache = _scan(SKILLS_DIR)
    return _cache


def list_skill_summaries(directory: Path | None = None) -> list[Skill]:
    return list(load_all_skills(directory).values())


def skill_catalog(directory: Path | None = None) -> str:
    """`name: description` lines for the system prompt.

    Without this the model has to spend a tool call on `list_skills` just to
    learn that a skill covering its question exists — and often does not
    bother, answering from the default prompt instead.
    """
    skills = list_skill_summaries(directory)
    if not skills:
        return ""
    lines = [
        f"- {skill.name}: {skill.description}" for skill in skills if skill.description
    ]
    if not lines:
        return ""
    return "Installed skills (call load_skill with the name):\n" + "\n".join(lines)


def get_skill(name: str, directory: Path | None = None) -> Skill:
    skills = load_all_skills(directory)
    skill = skills.get(name.strip())
    if skill is None:
        known = ", ".join(sorted(skills)) or "(none)"
        raise SkillError(f"Unknown skill {name!r}. Available: {known}.")
    return skill
