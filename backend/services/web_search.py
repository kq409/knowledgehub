"""DeepSeek Responses API web search for Ask's corrective fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

DUMMY_KEYS = frozenset({"", "ollama", "lm-studio", "changeme", "none"})


@dataclass(frozen=True)
class ExternalHit:
    url: str
    title: str
    snippet: str


class WebSearcher(Protocol):
    def available(self) -> bool: ...

    def search(self, query: str) -> list[ExternalHit]: ...


class WebSearchError(Exception):
    """Raised when the provider cannot complete a search."""


def web_search_configured() -> bool:
    key = (os.getenv("WEB_SEARCH_API_KEY") or "").strip()
    return bool(key) and key.lower() not in DUMMY_KEYS


def web_search_base_url() -> str:
    return (os.getenv("WEB_SEARCH_BASE_URL") or "https://api.deepseek.com").rstrip("/")


def web_search_model() -> str:
    return os.getenv("WEB_SEARCH_MODEL", "deepseek-v4-flash").strip() or (
        "deepseek-v4-flash"
    )


def _as_mapping(value: object) -> dict:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except TypeError:
            dumped = dump(mode="python")
        if isinstance(dumped, dict):
            return dumped
    return {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def parse_web_search_output(response: object) -> list[ExternalHit]:
    """Pull title/url/snippet from Responses API output, never the final prose."""
    hits: list[ExternalHit] = []
    seen: set[str] = set()

    def add(url: object, title: object = "", snippet: object = "") -> None:
        href = str(url or "").strip()
        if not href.startswith("http") or href in seen:
            return
        seen.add(href)
        label = str(title or "").strip() or href
        body = str(snippet or "").strip()
        hits.append(ExternalHit(url=href, title=label[:500], snippet=body[:800]))

    def walk(node: object) -> None:
        mapping = _as_mapping(node)
        if mapping:
            url = mapping.get("url") or mapping.get("uri") or mapping.get("link")
            if url:
                add(
                    url,
                    mapping.get("title") or mapping.get("name"),
                    mapping.get("snippet")
                    or mapping.get("text")
                    or mapping.get("excerpt"),
                )
            for nested in mapping.values():
                walk(nested)
            return
        if isinstance(node, list | tuple):
            for item in node:
                walk(item)
            return
        annotations = _get(node, "annotations")
        if annotations:
            walk(annotations)
        content = _get(node, "content")
        if content:
            walk(content)
        output = _get(node, "output")
        if output:
            walk(output)
        action = _get(node, "action")
        if action:
            walk(action)

    walk(_get(response, "output") or response)
    return hits


class WebSearchService:
    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
    ):
        self._client = client
        self._model = model or web_search_model()

    def available(self) -> bool:
        return web_search_configured() or self._client is not None

    def _client_or_none(self) -> OpenAI | None:
        if self._client is not None:
            return self._client
        if not web_search_configured():
            return None
        return OpenAI(
            api_key=os.getenv("WEB_SEARCH_API_KEY"),
            base_url=web_search_base_url(),
        )

    def search(self, query: str) -> list[ExternalHit]:
        question = query.strip()
        if not question:
            return []
        client = self._client_or_none()
        if client is None:
            raise WebSearchError("Web search is not configured")
        try:
            response = client.responses.create(
                model=self._model,
                input=(
                    "Search the web for recent, citable sources about:\n"
                    f"{question}\n"
                    "Prefer papers, surveys, and official docs."
                ),
                tools=[{"type": "web_search"}],
                tool_choice={"type": "web_search"},
            )
        except Exception as exc:
            raise WebSearchError(str(exc)) from exc
        return parse_web_search_output(response)
