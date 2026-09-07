"""Optional lexical/dense reranker. Off unless RERANK_ENABLED is set."""

from __future__ import annotations

import os
import threading

_ranker = None
_ranker_lock = threading.Lock()
_ranker_failed = False

DEFAULT_MODEL = "ms-marco-TinyBERT-L-2-v2"
RERANK_CANDIDATES = 32


def rerank_enabled() -> bool:
    return os.getenv("RERANK_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def rerank_model_name() -> str:
    return os.getenv("RERANK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _cache_dir() -> str:
    return os.getenv("HF_HOME") or os.getenv("FLASHRANK_CACHE") or "/tmp/flashrank"


def _get_ranker():
    global _ranker, _ranker_failed
    if _ranker is not None or _ranker_failed:
        return _ranker
    with _ranker_lock:
        if _ranker is not None or _ranker_failed:
            return _ranker
        try:
            from flashrank import Ranker

            _ranker = Ranker(model_name=rerank_model_name(), cache_dir=_cache_dir())
        except Exception as exc:
            _ranker_failed = True
            print(f"⚠️  FlashRank unavailable, skipping rerank: {exc}", flush=True)
            return None
        return _ranker


def rerank_texts(query: str, texts: list[str]) -> list[tuple[int, float]]:
    """Return (original_index, score) best-first. Identity order on failure."""
    if not query.strip() or not texts:
        return [(index, 0.0) for index in range(len(texts))]
    ranker = _get_ranker()
    if ranker is None:
        return [(index, 0.0) for index in range(len(texts))]
    try:
        from flashrank import RerankRequest

        passages = [
            {"id": index, "text": text or "", "meta": {}}
            for index, text in enumerate(texts)
        ]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
    except Exception as exc:
        print(f"⚠️  FlashRank rerank failed: {exc}", flush=True)
        return [(index, 0.0) for index in range(len(texts))]

    ranked: list[tuple[int, float]] = []
    for item in results:
        if isinstance(item, dict):
            ranked.append((int(item.get("id", 0)), float(item.get("score", 0.0))))
            continue
        ranked.append(
            (
                int(getattr(item, "id", 0)),
                float(getattr(item, "score", 0.0)),
            )
        )
    if not ranked:
        return [(index, 0.0) for index in range(len(texts))]
    return ranked
