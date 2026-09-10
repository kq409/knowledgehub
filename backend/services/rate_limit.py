"""In-process sliding window. One replica, on purpose."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self, *, limit: int, window_seconds: float) -> None:
        self.limit = max(0, int(limit))
        self.window_seconds = max(1.0, float(window_seconds))
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        bucket = self._hits[key]
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    if request.client is None:
        return "unknown"
    return request.client.host or "unknown"


_chat_limiter: RateLimiter | None = None


def chat_limiter() -> RateLimiter:
    global _chat_limiter
    from services.settings import chat_rate_limit_per_hour

    if _chat_limiter is None:
        _chat_limiter = RateLimiter(
            limit=chat_rate_limit_per_hour(), window_seconds=3600.0
        )
    return _chat_limiter


def require_chat_capacity(request: Request) -> None:
    if not chat_limiter().allow(client_ip(request)):
        raise HTTPException(
            status_code=429,
            detail="Rate limit: this demo caps Ask/Chat per IP. Try again later.",
        )
