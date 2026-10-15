"""In-memory sliding-window limiter (no Redis). Fine for a single API process."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.core.config import get_settings
from app.core.security import try_decode_user_id

LOGIN_PATH = "/api/v1/auth/login"


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_epoch: int


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def hit(self, key: str, limit: int, window_seconds: float = 60.0) -> RateLimitResult:
        now = time.time()
        with self._lock:
            bucket = self._hits[key]
            cutoff = now - window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                reset_epoch = int(bucket[0] + window_seconds)
                return RateLimitResult(False, limit, 0, reset_epoch)
            bucket.append(now)
            remaining = max(0, limit - len(bucket))
            reset_epoch = int(bucket[0] + window_seconds)
            return RateLimitResult(True, limit, remaining, reset_epoch)


limiter = SlidingWindowLimiter()


def client_ip(request: Request, *, trust_forwarded: bool = False) -> str:
    """Client identity for rate limits.

    `X-Forwarded-For` is ignored unless `trust_forwarded` is true. Spoofed
    forwarded headers would otherwise rotate the login IP bucket.
    """
    if trust_forwarded:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def apply_rate_limit_headers(response: Response, result: RateLimitResult) -> None:
    response.headers["X-RateLimit-Limit"] = str(result.limit)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    response.headers["X-RateLimit-Reset"] = str(result.reset_epoch)


def _denied(result: RateLimitResult) -> JSONResponse:
    response = JSONResponse(
        status_code=429,
        content={"detail": "Too many requests"},
    )
    apply_rate_limit_headers(response, result)
    response.headers["Retry-After"] = str(max(1, result.reset_epoch - int(time.time())))
    return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        settings = get_settings()
        ip = client_ip(request, trust_forwarded=settings.trust_x_forwarded_for)
        results: list[RateLimitResult] = []

        ip_result = limiter.hit(f"ip:{ip}", settings.rate_limit_per_minute)
        results.append(ip_result)
        if not ip_result.allowed:
            return _denied(ip_result)

        if request.url.path.rstrip("/") == LOGIN_PATH and request.method == "POST":
            login_result = limiter.hit(f"login:{ip}", settings.rate_limit_login_per_minute)
            results.append(login_result)
            if not login_result.allowed:
                return _denied(login_result)

        user_id = try_decode_user_id(request.headers.get("authorization"))
        if user_id:
            user_result = limiter.hit(f"user:{user_id}", settings.rate_limit_user_per_minute)
            results.append(user_result)
            if not user_result.allowed:
                return _denied(user_result)

        response = await call_next(request)
        # Advertise the tightest remaining quota that applied to this request.
        tightest = min(results, key=lambda item: (item.remaining, -item.limit))
        apply_rate_limit_headers(response, tightest)
        return response
