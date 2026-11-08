from app.core.rate_limit import RateLimitMiddleware, _SlidingWindowLimiter


class _Request:
    def __init__(self, authorization):
        self.headers = {"authorization": authorization}


def test_bearer_tokens_get_distinct_buckets_even_with_the_same_jwt_header():
    middleware = object.__new__(RateLimitMiddleware)
    token_a = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLWEifQ.signature-a"
    token_b = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLWIifQ.signature-b"

    key_a = middleware._caller_key(_Request(f"Bearer {token_a}"))
    key_b = middleware._caller_key(_Request(f"Bearer {token_b}"))

    assert key_a != key_b
    assert key_a == middleware._caller_key(_Request(f"Bearer {token_a}"))
    assert token_a not in key_a


def test_rate_limit_buckets_do_not_share_capacity_between_bearer_tokens():
    limiter = _SlidingWindowLimiter()
    assert limiter.allow("user-a", 1, 60) == (True, 0)
    assert limiter.allow("user-a", 1, 60)[0] is False
    assert limiter.allow("user-b", 1, 60) == (True, 0)
