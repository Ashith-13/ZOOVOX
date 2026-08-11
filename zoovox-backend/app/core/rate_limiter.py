"""
Sliding-window rate limiter using in-process dict.
Production: replace with Redis-backed implementation (redis-py aioredis).
"""

import time
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import settings

_request_log: dict[str, deque] = defaultdict(deque)

EXEMPT_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = settings.RATE_LIMIT_WINDOW_SECONDS
        max_req = settings.RATE_LIMIT_REQUESTS

        dq = _request_log[client_ip]
        while dq and dq[0] < now - window:
            dq.popleft()

        if len(dq) >= max_req:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {max_req} requests per {window}s.",
                headers={"Retry-After": str(window)},
            )

        dq.append(now)
        return await call_next(request)
