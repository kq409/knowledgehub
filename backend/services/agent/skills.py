"""On-disk skills the agent loads into context on demand.

Skills live as markdown files under backend/skills/ with YAML frontmatter
(name, description) and a body. They are never injected into the default
system prompt — the model calls list_skills / load_skill when it needs them.
"""

from __future__ import annotations

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


def load_all_skills(directory: Path | None = None) -> dict[str, Skill]:
    root = directory if directory is not None else SKILLS_DIR
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


def list_skill_summaries(directory: Path | None = None) -> list[Skill]:
    return list(load_all_skills(directory).values())


def get_skill(name: str, directory: Path | None = None) -> Skill:
    skills = load_all_skills(directory)
    skill = skills.get(name.strip())
    if skill is None:
        known = ", ".join(sorted(skills)) or "(none)"
        raise SkillError(f"Unknown skill {name!r}. Available: {known}.")
    return skill
