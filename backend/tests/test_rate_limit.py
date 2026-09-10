from services.rate_limit import RateLimiter


def test_rate_limiter_unlimited_when_limit_zero():
    limiter = RateLimiter(limit=0, window_seconds=60)
    assert limiter.allow("a")
    assert limiter.allow("a")


def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("ip")
    assert limiter.allow("ip")
    assert not limiter.allow("ip")


def test_rate_limiter_keys_are_independent():
    limiter = RateLimiter(limit=1, window_seconds=60)
    assert limiter.allow("alice")
    assert limiter.allow("bob")
    assert not limiter.allow("alice")
