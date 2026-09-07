"""Shared helpers for OpenAI-compatible chat.completions calls.

Thinking models (e.g. qwen3 via Ollama, DeepSeek V4) often emit a long
`reasoning` / `reasoning_content` field and an empty `content`. On Ollama's
`/v1` endpoint the switch is `reasoning_effort`; DeepSeek accepts the same
field. Without it set to `none`, Ask/Compare/Chat can hit max_tokens while
still thinking and look like an empty answer.

Paths can override effort via dedicated env vars (ASK_REASONING_EFFORT, etc.)
that fall back to LLM_REASONING_EFFORT, then to a caller default.
"""

from __future__ import annotations

import os
from typing import Any


def effort_from_env(*names: str, default: str = "none") -> str:
    """First non-empty env among `names`, else `default`.

    A value that is only whitespace means "omit the field" (provider default).
    """
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        return raw.strip().lower()
    return default.strip().lower()


def reasoning_extra_body(effort: str | None = None) -> dict[str, Any]:
    """Body fields that control thinking on thinking-capable providers.

    Pass `effort` explicitly, or omit to read `LLM_REASONING_EFFORT` (default
    `none`). Empty string omits the field so the provider keeps its default.
    """
    if effort is None:
        effort = effort_from_env("LLM_REASONING_EFFORT", default="none")
    else:
        effort = effort.strip().lower()
    if not effort:
        return {}
    return {"reasoning_effort": effort}


def with_chat_extras(
    kwargs: dict[str, Any], *, effort: str | None = None
) -> dict[str, Any]:
    """Merge provider extras into a chat.completions.create kwargs dict.

    `effort=None` uses `LLM_REASONING_EFFORT`. Pass a string (including `""`)
    to force a path-specific value from `effort_from_env(...)`.
    """
    extras = reasoning_extra_body(effort)
    if not extras:
        return kwargs
    extra_body = dict(kwargs.get("extra_body") or {})
    for key, value in extras.items():
        extra_body.setdefault(key, value)
    return {**kwargs, "extra_body": extra_body}


def max_tokens_from_env(name: str, fallback: int) -> int:
    """Read a token ceiling; ignore missing or non-numeric values.

    Values below 256 are clamped up so tiny misconfigs do not silently
    truncate every completion.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return max(256, value) if value > 0 else fallback


def message_text(message: Any) -> str:
    """Plain assistant text from a chat completion message."""
    return (getattr(message, "content", None) or "").strip()
