import time
from collections import defaultdict
from typing import Callable, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class TokenBucket:
    def __init__(self, rate: int, per_seconds: int):
        self.rate = rate
        self.per = per_seconds
        self.allowance = rate
        self.last_check = time.time()

    def allow(self):
        current = time.time()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.per)
        if self.allowance > self.rate:
            self.allowance = self.rate
        if self.allowance < 1.0:
            return False
        self.allowance -= 1.0
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests: int = 60, per_seconds: int = 60):
        super().__init__(app)
        self.buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(requests, per_seconds)
        )

    async def dispatch(self, request, call_next: Callable):
        client_ip = request.client.host if request.client else "anon"
        if not self.buckets[client_ip].allow():
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        return await call_next(request)
