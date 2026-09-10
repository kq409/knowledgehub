"""Process-wide flags for demo vs local. Not a config framework."""

from __future__ import annotations

import os
import secrets


def env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def app_env() -> str:
    return (os.getenv("APP_ENV") or "dev").strip().lower()


def is_prod() -> bool:
    return app_env() in {"demo", "prod", "production"}


def demo_mode() -> bool:
    if os.getenv("DEMO_MODE") is not None and os.getenv("DEMO_MODE") != "":
        return env_flag("DEMO_MODE")
    return is_prod()


def whisper_enabled() -> bool:
    return env_flag("WHISPER_ENABLED", default=not is_prod())


def grobid_enabled() -> bool:
    if not env_flag("GROBID_ENABLED", default=not is_prod()):
        return False
    return bool((os.getenv("GROBID_URL") or "").strip())


def uploads_enabled() -> bool:
    if not demo_mode():
        return True
    return env_flag("DEMO_UPLOADS", default=False)


def git_sha() -> str:
    return (os.getenv("GIT_SHA") or os.getenv("FLY_IMAGE_REF") or "unknown")[:40]


def cors_origins() -> list[str]:
    raw = (os.getenv("CORS_ORIGINS") or "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    if is_prod():
        return []
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def cors_origin_regex() -> str | None:
    override = (os.getenv("CORS_ORIGIN_REGEX") or "").strip()
    if override:
        return override
    if is_prod():
        return None
    return r"https?://(localhost|127\.0\.0\.1)(:\d+)?"


def chat_rate_limit_per_hour() -> int:
    raw = (os.getenv("CHAT_RATE_LIMIT_PER_HOUR") or "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 30 if demo_mode() else 0


def eval_token() -> str:
    return (os.getenv("EVAL_TOKEN") or "").strip()


def eval_run_allowed(provided: str | None) -> bool:
    """Local: open. Demo: require EVAL_TOKEN, or deny if unset."""
    expected = eval_token()
    if expected:
        got = (provided or "").strip()
        if not got or len(got) != len(expected):
            return False
        return secrets.compare_digest(got, expected)
    return not demo_mode()


def demo_reset_token() -> str:
    return (os.getenv("DEMO_RESET_TOKEN") or "").strip()


def frontend_dist() -> str:
    return (os.getenv("FRONTEND_DIST") or "").strip()
