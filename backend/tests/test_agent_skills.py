from pathlib import Path
from unittest.mock import MagicMock

import pytest

from services.agent.skills import (
    SKILLS_DIR,
    SkillError,
    get_skill,
    list_skill_summaries,
    load_all_skills,
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
