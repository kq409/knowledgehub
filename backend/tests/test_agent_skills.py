from pathlib import Path
from unittest.mock import MagicMock

import pytest

from services.agent.skills import (
    SKILLS_DIR,
    SkillError,
    get_skill,
    list_skill_summaries,
    load_all_skills,
    reload_skills,
    skill_catalog,
)
from services.agent.tools import ToolContext, list_skills_tool, load_skill_tool


def test_ships_the_four_planned_skills():
    skills = load_all_skills()
    assert set(skills) >= {
        "cite-from-library",
        "compare-papers",
        "structured-note",
        "voice-cleanup",
        "connect-note",
    }
    assert SKILLS_DIR.is_dir()


def test_list_summaries_include_descriptions():
    summaries = list_skill_summaries()
    by_name = {skill.name: skill for skill in summaries}
    assert "cite" in by_name["cite-from-library"].description.lower()


def test_unknown_skill_raises():
    with pytest.raises(SkillError, match="Unknown skill"):
        get_skill("does-not-exist")


async def test_list_and_load_tools_return_artifact():
    ctx = ToolContext(session=MagicMock(), embeddings=MagicMock())
    listed = await list_skills_tool(ctx)
    assert "cite-from-library" in listed.content

    loaded = await load_skill_tool(ctx, name="compare-papers")
    assert loaded.artifact is not None
    assert loaded.artifact["kind"] == "skill"
    assert loaded.artifact["data"]["name"] == "compare-papers"
    assert "method" in loaded.content.lower()


async def test_load_skill_tool_rejects_unknown():
    ctx = ToolContext(session=MagicMock(), embeddings=MagicMock())
    from services.agent.tools import ToolError

    with pytest.raises(ToolError, match="Unknown skill"):
        await load_skill_tool(ctx, name="nope")


def test_custom_directory_is_scanned(tmp_path: Path):
    skill_file = tmp_path / "demo.md"
    skill_file.write_text(
        "---\nname: demo\ndescription: A demo skill.\n---\n\nBody here.\n",
        encoding="utf-8",
    )
    skills = load_all_skills(tmp_path)
    assert list(skills) == ["demo"]
    assert get_skill("demo", tmp_path).body == "Body here."


def test_the_catalog_lists_names_and_descriptions_but_no_bodies():
    """The prompt gets five lines; the bodies stay on disk until asked for."""
    catalog = skill_catalog()

    assert "cite-from-library" in catalog
    assert "compare-papers" in catalog
    assert "load_skill" in catalog
    longest_body = max(len(skill.body) for skill in list_skill_summaries())
    assert len(catalog) < longest_body


def test_the_catalog_is_empty_when_nothing_is_installed(tmp_path: Path):
    assert skill_catalog(tmp_path) == ""


def test_the_default_directory_is_scanned_once(monkeypatch: pytest.MonkeyPatch):
    """It used to be re-read on every list_skills and every load_skill."""
    import services.agent.skills as skills_module

    scans = {"count": 0}
    real_scan = skills_module._scan

    def counting_scan(root: Path):
        scans["count"] += 1
        return real_scan(root)

    monkeypatch.setattr(skills_module, "_scan", counting_scan)
    monkeypatch.setattr(skills_module, "_cache", None)

    for _ in range(4):
        load_all_skills()
        skill_catalog()
    assert scans["count"] == 1

    reload_skills()
    assert scans["count"] == 2


def test_the_agent_prompt_carries_the_catalog():
    """A skill the model cannot see is a skill it will not load."""
    from services.agent.loop import AGENT_PROMPT

    assert "Installed skills" not in AGENT_PROMPT

    composed = f"{AGENT_PROMPT}\n\n{skill_catalog()}"
    for skill in list_skill_summaries():
        assert f"- {skill.name}: " in composed
