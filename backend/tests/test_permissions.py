from services.agent.permissions import (
    DEFAULT_POLICY,
    Decision,
    PermissionPolicy,
    PermissionRule,
)
from services.agent.tools import TOOL_HANDLERS


def test_every_shipped_tool_is_allowed():
    """A new tool has to be written into the policy, not just registered.

    The rules are listed by hand precisely so this fails when someone adds a
    handler without deciding whether it may run.
    """
    assert sorted(TOOL_HANDLERS) == DEFAULT_POLICY.allowed_tools()
    for name in TOOL_HANDLERS:
        assert DEFAULT_POLICY.check(name).decision is Decision.allow


def test_the_comparison_tool_is_allowed_despite_saving_a_row():
    assert DEFAULT_POLICY.check("compare_papers").allowed
    assert DEFAULT_POLICY.check("preview_note_extraction").allowed
    assert DEFAULT_POLICY.check("todo_write").allowed
    assert DEFAULT_POLICY.check("spawn_subagent").allowed


def test_subagent_policy_excludes_spawn_and_compare():
    from services.agent.permissions import subagent_policy
    from services.agent.tools import SUBAGENT_TOOL_NAMES

    policy = subagent_policy()
    assert sorted(policy.allowed_tools()) == sorted(SUBAGENT_TOOL_NAMES)
    assert not policy.check("spawn_subagent").allowed
    assert not policy.check("compare_papers").allowed
    assert not policy.check("present_workspace").allowed
    assert not policy.check("todo_write").allowed
    assert not policy.check("memory_write").allowed
    assert not policy.check("memory_delete").allowed
    assert not policy.check("link_note").allowed
    assert not policy.check("unlink_note").allowed
    assert not policy.check("connect_note").allowed
    assert policy.check("memory_search").allowed
    assert policy.check("load_skill").allowed


def test_a_tool_nobody_registered_is_refused():
    verdict = DEFAULT_POLICY.check("delete_library")

    assert verdict.decision is Decision.deny
    assert not verdict.allowed
    assert "delete_library" in verdict.reason


def test_an_allowed_verdict_carries_no_complaint():
    verdict = DEFAULT_POLICY.check("search_library")

    assert verdict.allowed
    assert verdict.reason == ""


def test_ask_is_not_allowed_while_the_channel_cannot_ask():
    policy = PermissionPolicy(
        [PermissionRule(tool="publish_note", decision=Decision.ask)]
    )

    verdict = policy.check("publish_note")

    assert verdict.decision is Decision.ask
    assert not verdict.allowed
    assert "approval" in verdict.reason


def test_a_rule_may_state_its_own_reason():
    policy = PermissionPolicy(
        [
            PermissionRule(
                tool="fetch_web",
                decision=Decision.deny,
                reason="reaches outside the library",
            )
        ]
    )

    assert policy.check("fetch_web").reason == "reaches outside the library"
