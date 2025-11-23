import logging
import os

from redis.asyncio import from_url as redis_from_url
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app, requests: int = 60, per_seconds: int = 60, redis_url: str = None
    ):
        super().__init__(app)
        self.requests = requests
        self.per = per_seconds
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.r = redis_from_url(self.redis_url, decode_responses=True)

    async def dispatch(self, request, call_next):
        client_ip = "unknown"
        try:
            client_ip = request.client.host if request.client else "anon"
            key = f"ratelimit:{client_ip}"
            cur = await self.r.incr(key)
            if cur == 1:
                await self.r.expire(key, self.per)
            if cur > self.requests:
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        except Exception as e:
            log.warning(f"Rate limit check failed for {client_ip}: {e}")
        return await call_next(request)
