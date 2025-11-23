"""
Simple file-based cache implementation for development
Replaces Redis functionality when Redis is not available
"""

import json
import time
from pathlib import Path
from typing import Optional


class FileCache:
    """Simple file-based cache for development environments"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key"""
        # Create a safe filename from key
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-").strip()
        if not safe_key:
            safe_key = "default"
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check expiration
            if time.time() > data.get("expires_at", 0):
                cache_path.unlink()  # Remove expired cache
                return None

            return data.get("value")

        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def set(self, key: str, value: str, expire_seconds: int = 3600) -> bool:
        """Set value in cache with expiration"""
        cache_path = self._get_cache_path(key)

        data = {
            "value": value,
            "expires_at": time.time() + expire_seconds,
            "created_at": time.time(),
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            return True
        except OSError:
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        cache_path = self._get_cache_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()
            return True
        except OSError:
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None

    def clear(self) -> bool:
        """Clear all cache files"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            return True
        except OSError:
            return False


# Global cache instance
file_cache = FileCache()


def get_cache():
    """Get cache instance (file-based for development)"""
    return file_cache


# Convenience functions for Redis-like interface
def cache_get(key: str) -> Optional[str]:
    """Get value from cache"""
    return file_cache.get(key)


def cache_set(key: str, value: str, expire_seconds: int = 3600) -> bool:
    """Set value in cache"""
    return file_cache.set(key, value, expire_seconds)


def cache_delete(key: str) -> bool:
    """Delete key from cache"""
    return file_cache.delete(key)


def cache_exists(key: str) -> bool:
    """Check if key exists"""
    return file_cache.exists(key)
