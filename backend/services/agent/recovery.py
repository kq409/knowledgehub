"""What to do when the model call itself fails.

Before this existed, a single transient 429 from DeepSeek or a hiccup from a
local Ollama ended the whole turn: the exception travelled up through the SSE
generator and the researcher lost the answer along with every tool result that
had already been paid for.

Four failures are worth recovering from, in increasing order of how much they
cost to handle:

    rate limited / overloaded   wait and try the same request again
    output truncated            ask for the same thing with more room
    context too long            summarise the history, then try again
    anything else               hand it to the caller

The classification is deliberately string-based as well as type-based:
"OpenAI-compatible" covers Ollama, LM Studio, vLLM and DeepSeek, and they do
not agree on which exception class carries a 429.
"""

from __future__ import annotations

import os
from enum import Enum

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "server_busy",
    "temporarily unavailable",
    "service unavailable",
    "capacity",
)

_CONTEXT_MARKERS = (
    "prompt_too_long",
    "context_length_exceeded",
    "too many tokens",
    "maximum context length",
    "context window",
    "reduce the length of the messages",
    "input length and `max_tokens` exceed",
    "exceeds the maximum",
)


class LlmFailure(str, Enum):
    rate_limited = "rate_limited"
    context_too_long = "context_too_long"
    fatal = "fatal"


def _status_code(error: BaseException) -> int | None:
    for attribute in ("status_code", "http_status", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def classify_llm_failure(error: BaseException) -> LlmFailure:
    """Which recovery, if any, this error is eligible for."""
    text = str(error).lower()

    # Context length first: some providers report it as a 400 and others fold
    # it into a generic server error, but the message is always explicit.
    if any(marker in text for marker in _CONTEXT_MARKERS):
        return LlmFailure.context_too_long

    if _status_code(error) in RETRYABLE_STATUS:
        return LlmFailure.rate_limited
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return LlmFailure.rate_limited

    return LlmFailure.fatal


def max_retries() -> int:
    """How many times a rate-limited call is retried before giving up."""
    try:
        return max(0, int(os.getenv("AGENT_LLM_RETRIES", "3")))
    except ValueError:
        return 3


def backoff_seconds(attempt: int, *, base: float = 0.5, cap: float = 8.0) -> float:
    """Exponential, capped. `attempt` is 0-based."""
    return min(cap, base * (2**attempt))
