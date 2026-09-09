"""Structured logging for the agent harness.

Every call site passes fields, not a formatted sentence, so the same event can
be read by a human in the devcontainer terminal and by a log pipeline in a
deployment. `AGENT_LOG_FORMAT=json` switches the rendering; the fields are the
same either way, which is what makes them worth attaching in the first place.

Two fields are always present when the caller has them: `run_id` ties every
line to one `ResearchAgent.run` call, and `agent_id` separates the main loop
from a nested pass. Anything else is event-specific.
"""

from __future__ import annotations

import json
import logging
import os

LOGGER_NAME = "knowledgehub.agent"
FIELDS_ATTR = "agent_fields"

# Attributes `logging` puts on every record. Anything else on a record came
# from an `extra=` and belongs in the JSON payload.
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | frozenset({"message", "asctime", "taskName", FIELDS_ATTR})

logger = logging.getLogger(LOGGER_NAME)


def log_event(event: str, level: int = logging.INFO, **fields: object) -> None:
    """Record one harness event with machine-readable fields."""
    logger.log(level, event, extra={FIELDS_ATTR: fields})


def log_warning(event: str, **fields: object) -> None:
    log_event(event, level=logging.WARNING, **fields)


def _record_fields(record: logging.LogRecord) -> dict:
    fields = dict(getattr(record, FIELDS_ATTR, None) or {})
    for key, value in record.__dict__.items():
        if key not in _RESERVED and key not in fields:
            fields[key] = value
    return fields


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for a log pipeline."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(_record_fields(record))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """`event key=value` — readable in the devcontainer terminal."""

    def format(self, record: logging.LogRecord) -> str:
        fields = _record_fields(record)
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        line = f"{record.levelname[0]} {record.getMessage()}"
        if rendered:
            line = f"{line} {rendered}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_agent_logging(fmt: str | None = None) -> None:
    """Install one handler on the agent logger. Safe to call twice."""
    chosen = (fmt if fmt is not None else os.getenv("AGENT_LOG_FORMAT") or "").strip()
    formatter = JsonFormatter() if chosen.lower() == "json" else TextFormatter()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.addHandler(handler)
    logger.setLevel(os.getenv("AGENT_LOG_LEVEL", "INFO").upper())
    # Uvicorn already owns the root handler; ours would print every line twice.
    logger.propagate = False
