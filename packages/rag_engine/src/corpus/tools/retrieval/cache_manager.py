"""
Redis-based query cache manager with intelligent invalidation.
Standalone: works with or without Redis (graceful degradation).
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.common.paths import SEARCH_CACHE_DIR

logger = logging.getLogger(__name__)

# Allow disabling Redis via environment variable (fast local fallback)
_DISABLE_REDIS = os.getenv("RAG_DISABLE_REDIS", "").strip() in {
    "1",
    "true",
    "yes",
    "on",
}

try:
    if not _DISABLE_REDIS:
        import redis

        REDIS_AVAILABLE = True
    else:
        raise ImportError("Redis disabled by RAG_DISABLE_REDIS env var")
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning(
        "Redis disabled or not installed. Using local cache. Set RAG_DISABLE_REDIS=0 to enable."
    )


class QueryCache:
    """
    Intelligent query result caching with TTL and invalidation.
    Falls back to in-memory cache if Redis unavailable.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        default_ttl: int = 3600,
        enable_local_fallback: bool = True,
    ):
        """
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            default_ttl: Default time-to-live in seconds (1 hour)
            enable_local_fallback: Use in-memory cache if Redis unavailable
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.default_ttl = default_ttl
        self.enable_local_fallback = enable_local_fallback

        self.redis_client = None
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.using_local = False

        # Try to connect to Redis unless disabled
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False,
                    socket_connect_timeout=2,
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"[+] Connected to Redis at {redis_host}:{redis_port}")
            except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
                logger.warning(f"[!]️  Redis connection failed: {e}")
                self.redis_client = None
                if enable_local_fallback:
                    logger.info("📦 Using in-memory cache fallback")
                    self.using_local = True
                else:
                    raise
        else:
            # Explicitly disabled -> use local
            if enable_local_fallback:
                logger.info("📦 Using in-memory cache (Redis disabled)")
                self.using_local = True

    def _make_key(self, query: str, top_k: int, filters: Optional[Dict] = None) -> str:
        """Generate cache key from query parameters."""
        key_data = {
            "query": query.lower().strip(),
            "top_k": top_k,
            "filters": filters or {},
        }
        key_str = json.dumps(key_data, sort_keys=True)
        hash_val = hashlib.sha256(key_str.encode()).hexdigest()
        return f"rag:query:{hash_val}"

    def get(
        self, query: str, top_k: int, filters: Optional[Dict] = None
    ) -> Optional[List[Dict]]:
        """
        Retrieve cached results if available and not expired.

        Returns:
            List of results or None if not in cache/expired
        """
        key = self._make_key(query, top_k, filters)

        try:
            if self.redis_client:
                # Try Redis first
                cached = self.redis_client.get(key)
                if cached:
                    # Decode from JSON instead of pickle for security
                    results = json.loads(
                        cached.decode("utf-8") if isinstance(cached, bytes) else cached
                    )
                    logger.debug(f"🎯 Cache hit (Redis): {key[:20]}...")
                    return results
            elif self.using_local:
                # Use local cache
                if key in self.local_cache:
                    entry = self.local_cache[key]
                    if entry["expires_at"] > datetime.now():
                        logger.debug(f"🎯 Cache hit (Local): {key[:20]}...")
                        return entry["results"]
                    else:
                        # Expired, remove
                        del self.local_cache[key]
        except Exception as e:
            logger.warning(f"[!]️  Cache get error: {e}")

        return None

    def set(
        self,
        query: str,
        top_k: int,
        results: List[Dict],
        ttl: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> bool:
        """
        Cache results with TTL.

        Args:
            query: Search query
            top_k: Number of results
            results: Search results to cache
            ttl: Time-to-live in seconds (uses default if None)
            filters: Query filters for cache key

        Returns:
            True if cached successfully
        """
        key = self._make_key(query, top_k, filters)
        ttl = ttl or self.default_ttl

        try:
            if self.redis_client:
                # Use JSON instead of pickle for security
                cached_data = json.dumps(results)
                self.redis_client.setex(key, ttl, cached_data)
                logger.debug(f"[save] Cached (Redis): {key[:20]}... TTL={ttl}s")
                return True
            elif self.using_local:
                self.local_cache[key] = {
                    "results": results,
                    "expires_at": datetime.now() + timedelta(seconds=ttl),
                }
                logger.debug(f"[save] Cached (Local): {key[:20]}... TTL={ttl}s")
                return True
        except Exception as e:
            logger.warning(f"[!]️  Cache set error: {e}")
            return False

        return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Redis key pattern (e.g., "rag:query:*")

        Returns:
            Number of keys deleted
        """
        deleted = 0
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.info(f"🗑️  Invalidated {deleted} cache entries")
            elif self.using_local:
                # For local cache, match by prefix
                prefix = pattern.replace("*", "")
                keys_to_delete = [
                    k for k in self.local_cache.keys() if k.startswith(prefix)
                ]
                for k in keys_to_delete:
                    del self.local_cache[k]
                deleted = len(keys_to_delete)
                logger.info(f"🗑️  Invalidated {deleted} local cache entries")
        except Exception as e:
            logger.warning(f"[!]️  Invalidation error: {e}")

        return deleted

    def invalidate_all(self) -> bool:
        """Clear entire cache."""
        try:
            if self.redis_client:
                self.redis_client.flushdb()
                logger.info("🗑️  Redis cache cleared")
            elif self.using_local:
                self.local_cache.clear()
                logger.info("🗑️  Local cache cleared")
            return True
        except Exception as e:
            logger.warning(f"[!]️  Clear cache error: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backend": (
                "redis"
                if self.redis_client
                else "local" if self.using_local else "none"
            ),
            "ttl": self.default_ttl,
        }

        try:
            if self.redis_client:
                info = self.redis_client.info()
                stats["used_memory"] = info.get("used_memory_human", "N/A")
                stats["connected_clients"] = info.get("connected_clients", 0)
                keys = self.redis_client.dbsize()
                stats["cached_queries"] = keys
            elif self.using_local:
                stats["cached_queries"] = len(self.local_cache)
                stats["memory_entries"] = len(self.local_cache)
        except Exception as e:
            logger.warning(f"[!]️  Stats error: {e}")

        return stats

    def health_check(self) -> bool:
        """Check if cache is available and healthy."""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
            elif self.using_local:
                return True
        except Exception as e:
            logger.error(f"[X] Cache health check failed: {e}")
        return False


# Singleton instance
_cache_instance: Optional[QueryCache] = None


def get_cache(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    default_ttl: int = 3600,
) -> QueryCache:
    """Get or create singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            default_ttl=default_ttl,
        )
    return _cache_instance
