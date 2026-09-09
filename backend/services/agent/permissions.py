"""What the agent is allowed to run.

Every tool call passes through a policy before it executes. Read tools are
allowed; write tools ask the researcher unless `ASK_APPROVAL_MODE=off`.
Unknown tools are refused. Listing each tool by hand means a new write or an
outside-the-library call has to be argued for here first.
"""

import os
from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    allow = "allow"
    deny = "deny"
    ask = "ask"


@dataclass(frozen=True)
class PermissionVerdict:
    decision: Decision
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.allow


@dataclass(frozen=True)
class PermissionRule:
    tool: str
    decision: Decision
    reason: str = ""


UNKNOWN_TOOL_REASON = "is not a tool this agent has"
APPROVAL_REASON = "needs the researcher's approval before it can change the library"
# Kept as an alias so older tests and messages that mention the unsupported
# channel still match. The channel can ask now; the string stays honest when
# no approver is wired up.
APPROVAL_UNSUPPORTED_REASON = APPROVAL_REASON

WRITE_TOOLS = frozenset(
    {
        "memory_write",
        "memory_delete",
        "link_note",
        "unlink_note",
        "connect_note",
    }
)

_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def approval_enabled() -> bool:
    """Whether write tools pause for a human.

    Default on: a mutating call in an enterprise setting should not go
    through because nobody was looking. Tests set `ASK_APPROVAL_MODE=off`
    so the rest of the suite does not hang waiting for a click.
    """
    raw = (os.getenv("ASK_APPROVAL_MODE") or "on").strip().lower()
    return raw not in _OFF_VALUES


class PermissionPolicy:
    """Decides whether a named tool may run.

    `ask` is a real decision. The hook chain either pauses for a human or,
    when approval is switched off, treats it as allow so existing behaviour
    is one env var away.
    """

    def __init__(self, rules: list[PermissionRule]) -> None:
        self._rules = {rule.tool: rule for rule in rules}

    def check(self, tool_name: str) -> PermissionVerdict:
        rule = self._rules.get(tool_name)
        if rule is None:
            return PermissionVerdict(
                decision=Decision.deny,
                reason=f"{tool_name!r} {UNKNOWN_TOOL_REASON}",
            )
        if rule.decision is Decision.ask:
            return PermissionVerdict(
                decision=Decision.ask,
                reason=rule.reason or f"{tool_name!r} {APPROVAL_UNSUPPORTED_REASON}",
            )
        return PermissionVerdict(decision=rule.decision, reason=rule.reason)

    def allowed_tools(self) -> list[str]:
        return sorted(
            name
            for name, rule in self._rules.items()
            if rule.decision is Decision.allow
        )

    def known_tools(self) -> list[str]:
        return sorted(self._rules)

    def rules(self) -> list[PermissionRule]:
        return list(self._rules.values())


SHIPPED_RULES = [
    PermissionRule(tool="search_library", decision=Decision.allow),
    # Public web via DeepSeek Responses API. Not a library tool; needs a real
    # WEB_SEARCH_API_KEY. Subagents and MCP do not get this.
    PermissionRule(tool="web_search", decision=Decision.allow),
    PermissionRule(tool="list_papers", decision=Decision.allow),
    PermissionRule(tool="list_notes", decision=Decision.allow),
    PermissionRule(tool="list_documents", decision=Decision.allow),
    PermissionRule(tool="read_paper", decision=Decision.allow),
    PermissionRule(tool="read_note", decision=Decision.allow),
    PermissionRule(tool="read_document", decision=Decision.allow),
    # Saves a derived comparison row, but never touches a paper or a note. The
    # researcher can delete it from the Compare tab like any other comparison.
    PermissionRule(tool="compare_papers", decision=Decision.allow),
    # Opens an interactive UI the researcher operates; does not mutate data.
    PermissionRule(tool="present_workspace", decision=Decision.allow),
    # Runs the note extraction model without saving anything.
    PermissionRule(tool="preview_note_extraction", decision=Decision.allow),
    PermissionRule(tool="todo_write", decision=Decision.allow),
    PermissionRule(tool="spawn_subagent", decision=Decision.allow),
    # Reads back a result this same turn already produced and truncated. It
    # cannot reach anything the turn has not already been allowed to see.
    PermissionRule(tool="fetch_tool_result", decision=Decision.allow),
    PermissionRule(tool="list_skills", decision=Decision.allow),
    PermissionRule(tool="load_skill", decision=Decision.allow),
    PermissionRule(tool="memory_search", decision=Decision.allow),
    PermissionRule(tool="memory_write", decision=Decision.ask, reason=APPROVAL_REASON),
    PermissionRule(tool="memory_delete", decision=Decision.ask, reason=APPROVAL_REASON),
    PermissionRule(tool="link_note", decision=Decision.ask, reason=APPROVAL_REASON),
    PermissionRule(tool="unlink_note", decision=Decision.ask, reason=APPROVAL_REASON),
    PermissionRule(tool="connect_note", decision=Decision.ask, reason=APPROVAL_REASON),
]


def subagent_policy() -> PermissionPolicy:
    """Read-only research tools only — no spawn, compare, preview, or todos."""
    from services.agent.tools import SUBAGENT_TOOL_NAMES

    return PermissionPolicy(
        [
            PermissionRule(tool=name, decision=Decision.allow)
            for name in sorted(SUBAGENT_TOOL_NAMES)
        ]
    )


def default_policy() -> PermissionPolicy:
    """The rules the agent ships with, written out one tool at a time.

    Deriving these from TOOL_HANDLERS would be shorter, but then every new tool
    would let itself in. Listing them means a tool that writes, deletes, or
    reaches outside the library has to be argued for here first.
    """
    return PermissionPolicy(SHIPPED_RULES)


DEFAULT_POLICY = default_policy()
