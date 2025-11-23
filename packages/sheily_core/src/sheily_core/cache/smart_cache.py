"""
Smart Multi-Level Cache System for Sheily AI
Provides intelligent caching with semantic search and performance optimization
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl: Optional[int] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl)

    def touch(self):
        """Update access metadata"""
        self.accessed_at = datetime.now()
        self.access_count += 1


class SemanticCache:
    """Semantic cache using embeddings for similarity search"""

    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.85):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.embeddings_cache: Dict[str, List[float]] = {}

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text (simplified version)"""
        # In production, this would use actual embeddings
        # For now, use simple hash-based similarity
        hash_obj = hashlib.md5(text.encode())
        # Convert hash to list of floats between 0 and 1
        return [int(b) / 255.0 for b in hash_obj.digest()[:16]]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

    async def find_similar(
        self, query: str, top_k: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Find similar cached entries using semantic similarity"""
        try:
            query_embedding = self._get_embedding(query)

            best_matches = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    continue

                if key not in self.embeddings_cache:
                    self.embeddings_cache[key] = self._get_embedding(key)

                similarity = self._cosine_similarity(
                    query_embedding, self.embeddings_cache[key]
                )

                if similarity >= self.similarity_threshold:
                    best_matches.append(
                        {"key": key, "entry": entry, "similarity": similarity}
                    )

            # Sort by similarity and return top match
            best_matches.sort(key=lambda x: x["similarity"], reverse=True)

            if best_matches:
                entry = best_matches[0]["entry"]
                entry.touch()
                return {
                    "response": entry.value,
                    "similarity": best_matches[0]["similarity"],
                    "cached_at": entry.created_at.isoformat(),
                }

            return None

        except Exception as e:
            logger.warning(f"Error in semantic cache search: {e}")
            return None

    async def store(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store entry in semantic cache"""
        try:
            # Evict old entries if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                ttl=ttl,
            )

            self.cache[key] = entry
            self.embeddings_cache[key] = self._get_embedding(key)

        except Exception as e:
            logger.error(f"Error storing in semantic cache: {e}")

    def _evict_oldest(self):
        """Evict least recently used entries"""
        if not self.cache:
            return

        # Sort by access time and remove oldest
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].accessed_at)

        # Remove 10% of oldest entries
        to_remove = max(1, len(sorted_entries) // 10)
        for i in range(to_remove):
            key, _ = sorted_entries[i]
            del self.cache[key]
            self.embeddings_cache.pop(key, None)


class ResponseCache:
    """Exact response cache for identical queries"""

    def __init__(self, max_size: int = 5000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size

    def _make_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Create cache key from query and parameters"""
        key_data = {"query": query}
        if params:
            # Sort params for consistent keys
            key_data["params"] = json.dumps(params, sort_keys=True)
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def get_exact(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Get exact match from cache"""
        try:
            key = self._make_key(query, params)

            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.touch()
                    return entry.value
                else:
                    # Remove expired entry
                    del self.cache[key]

            return None

        except Exception as e:
            logger.warning(f"Error getting exact match from cache: {e}")
            return None

    async def store_exact(
        self,
        query: str,
        response: Any,
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ):
        """Store exact response in cache"""
        try:
            key = self._make_key(query, params)

            # Evict if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            entry = CacheEntry(
                key=key,
                value=response,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                ttl=ttl,
            )

            self.cache[key] = entry

        except Exception as e:
            logger.error(f"Error storing exact response in cache: {e}")

    def _evict_lru(self):
        """Evict least recently used entries"""
        if not self.cache:
            return

        # Find entry with oldest access time
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].accessed_at)

        del self.cache[oldest_key]


class SmartCache:
    """
    Intelligent multi-level cache system

    Provides three levels of caching:
    1. Semantic cache - finds similar queries using embeddings
    2. Exact cache - exact query matches
    3. Fallback - passes through to original function
    """

    def __init__(
        self,
        semantic_cache_size: int = 1000,
        exact_cache_size: int = 5000,
        semantic_threshold: float = 0.85,
        default_ttl: int = 3600,
    ):
        self.semantic_cache = SemanticCache(
            max_size=semantic_cache_size, similarity_threshold=semantic_threshold
        )
        self.exact_cache = ResponseCache(max_size=exact_cache_size)
        self.default_ttl = default_ttl
        self.stats = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,
            "exact_hits": 0,
            "errors": 0,
        }

    async def get_or_compute(
        self,
        query: str,
        compute_func: Callable[..., Any],
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get from cache or compute and cache result

        Args:
            query: The query string
            compute_func: Function to compute result if not cached
            params: Additional parameters for the query
            ttl: Time to live in seconds

        Returns:
            Cached or computed result
        """
        try:
            # Try exact match first (fastest)
            exact_result = await self.exact_cache.get_exact(query, params)
            if exact_result is not None:
                self.stats["hits"] += 1
                self.stats["exact_hits"] += 1
                logger.debug(f"Cache hit (exact): {query[:50]}...")
                return exact_result

            # Try semantic match (slower but smarter)
            semantic_result = await self.semantic_cache.find_similar(query)
            if semantic_result is not None:
                self.stats["hits"] += 1
                self.stats["semantic_hits"] += 1
                logger.debug(
                    f"Cache hit (semantic): {query[:50]}... (similarity: {semantic_result['similarity']:.2f})"
                )
                return semantic_result["response"]

            # Cache miss - compute result
            self.stats["misses"] += 1
            logger.debug(f"Cache miss: {query[:50]}...")

            # Compute result
            if asyncio.iscoroutinefunction(compute_func):
                result = await compute_func(query, **(params or {}))
            else:
                result = compute_func(query, **(params or {}))

            # Cache the result
            cache_ttl = ttl or self.default_ttl
            await self.exact_cache.store_exact(query, result, params, cache_ttl)
            await self.semantic_cache.store(query, result, cache_ttl)

            return result

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in smart cache: {e}")
            # Fallback to computation without caching
            if asyncio.iscoroutinefunction(compute_func):
                return await compute_func(query, **(params or {}))
            else:
                return compute_func(query, **(params or {}))

    async def get(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Get from cache only (no computation)"""
        try:
            # Try exact match first
            exact_result = await self.exact_cache.get_exact(query, params)
            if exact_result is not None:
                self.stats["exact_hits"] += 1
                return exact_result

            # Try semantic match
            semantic_result = await self.semantic_cache.find_similar(query)
            if semantic_result is not None:
                self.stats["semantic_hits"] += 1
                return semantic_result["response"]

            return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    async def set(
        self,
        query: str,
        value: Any,
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ):
        """Manually set cache entry"""
        try:
            cache_ttl = ttl or self.default_ttl
            await self.exact_cache.store_exact(query, value, params, cache_ttl)
            await self.semantic_cache.store(query, value, cache_ttl)
        except Exception as e:
            logger.error(f"Error setting cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests) if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "semantic_hits": self.stats["semantic_hits"],
            "exact_hits": self.stats["exact_hits"],
            "errors": self.stats["errors"],
            "semantic_cache_size": len(self.semantic_cache.cache),
            "exact_cache_size": len(self.exact_cache.cache),
        }

    async def clear(self):
        """Clear all caches"""
        try:
            self.semantic_cache.cache.clear()
            self.semantic_cache.embeddings_cache.clear()
            self.exact_cache.cache.clear()
            logger.info("All caches cleared")
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")

    async def cleanup_expired(self):
        """Remove expired entries"""
        try:
            # Semantic cache cleanup
            expired_keys = [
                key
                for key, entry in self.semantic_cache.cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self.semantic_cache.cache[key]
                self.semantic_cache.embeddings_cache.pop(key, None)

            # Exact cache cleanup
            expired_keys = [
                key
                for key, entry in self.exact_cache.cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self.exact_cache.cache[key]

            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global cache instance
_smart_cache: Optional[SmartCache] = None


def get_smart_cache() -> SmartCache:
    """Get global smart cache instance"""
    global _smart_cache
    if _smart_cache is None:
        _smart_cache = SmartCache()
    return _smart_cache


# Convenience functions for easy integration
async def cached_query(
    query: str,
    compute_func: Callable[..., Any],
    params: Optional[Dict[str, Any]] = None,
    ttl: Optional[int] = None,
) -> Any:
    """Convenience function for cached queries"""
    cache = get_smart_cache()
    return await cache.get_or_compute(query, compute_func, params, ttl)


async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    cache = get_smart_cache()
    return cache.get_stats()


# Example usage in existing code:
"""
# Before (without cache)
response = await llm_service.generate(query, **params)

# After (with smart cache)
from sheily_core.cache.smart_cache import cached_query

response = await cached_query(
    query=query,
    compute_func=llm_service.generate,
    params=params,
    ttl=1800  # 30 minutes
)
"""
