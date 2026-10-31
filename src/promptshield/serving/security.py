"""API-key auth and a per-key sliding-window rate limiter."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from promptshield.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> str:
    if not key or not any(hmac.compare_digest(key, k) for k in settings.api_key_set):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")
    return key


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded", headers={"Retry-After": "60"}
                )
            q.append(now)


limiter = RateLimiter(settings.rate_limit_per_minute)


def authorized(key: str = Depends(require_api_key)) -> str:
    limiter.check(key)
    return key
