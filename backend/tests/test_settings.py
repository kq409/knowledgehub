from services.settings import (
    chat_rate_limit_per_hour,
    cors_origin_regex,
    cors_origins,
    demo_mode,
    eval_run_allowed,
    grobid_enabled,
    uploads_enabled,
    whisper_enabled,
)


def test_demo_mode_follows_app_env(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("APP_ENV", "demo")
    assert demo_mode() is True
    monkeypatch.setenv("APP_ENV", "dev")
    assert demo_mode() is False


def test_demo_mode_explicit_override(monkeypatch):
    monkeypatch.setenv("APP_ENV", "demo")
    monkeypatch.setenv("DEMO_MODE", "false")
    assert demo_mode() is False


def test_whisper_and_grobid_default_off_in_prod(monkeypatch):
    monkeypatch.setenv("APP_ENV", "demo")
    monkeypatch.delenv("WHISPER_ENABLED", raising=False)
    monkeypatch.delenv("GROBID_ENABLED", raising=False)
    monkeypatch.delenv("GROBID_URL", raising=False)
    assert whisper_enabled() is False
    assert grobid_enabled() is False


def test_uploads_locked_in_demo(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("DEMO_UPLOADS", raising=False)
    assert uploads_enabled() is False
    monkeypatch.setenv("DEMO_UPLOADS", "true")
    assert uploads_enabled() is True


def test_eval_locked_in_demo_without_token(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("EVAL_TOKEN", raising=False)
    assert eval_run_allowed(None) is False
    monkeypatch.setenv("EVAL_TOKEN", "secret")
    assert eval_run_allowed("secret") is True
    assert eval_run_allowed("nope") is False


def test_chat_rate_limit_defaults(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("CHAT_RATE_LIMIT_PER_HOUR", raising=False)
    assert chat_rate_limit_per_hour() == 30
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("APP_ENV", "dev")
    assert chat_rate_limit_per_hour() == 0


def test_prod_cors_has_no_localhost_regex(monkeypatch):
    monkeypatch.setenv("APP_ENV", "demo")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    monkeypatch.delenv("CORS_ORIGIN_REGEX", raising=False)
    assert cors_origins() == []
    assert cors_origin_regex() is None
