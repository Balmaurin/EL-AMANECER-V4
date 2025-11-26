"""
SISTEMA DE CACHE DISTRIBUIDO - SHEILY AI
"""

import json
from collections import OrderedDict
from datetime import datetime, timedelta


class DistributedCacheSystem:
    def __init__(self):
        self.local_cache = OrderedDict()
        self.max_local_size = 10000
        self.stats = {"hits": 0, "misses": 0, "sets": 0}

    def set(self, key: str, value, ttl: int = 3600):
        self.local_cache[key] = {
            "value": value,
            "expires": datetime.now() + timedelta(seconds=ttl),
        }
        if len(self.local_cache) > self.max_local_size:
            self.local_cache.popitem(last=False)
        self.stats["sets"] += 1
        return True

    def get(self, key: str):
        if key in self.local_cache:
            data = self.local_cache[key]
            if datetime.now() < data["expires"]:
                self.stats["hits"] += 1
                return data["value"]
            else:
                del self.local_cache[key]
        self.stats["misses"] += 1
        return None

    def get_stats(self):
        total = self.stats["hits"] + self.stats["misses"]
        return {
            "total_requests": total,
            "hit_rate": self.stats["hits"] / total if total > 0 else 0,
            "local_cache_size": len(self.local_cache),
        }


distributed_cache = DistributedCacheSystem()


def cache_set(key: str, value, ttl: int = 3600):
    return distributed_cache.set(key, value, ttl)


def cache_get(key: str):
    return distributed_cache.get(key)


def get_cache_stats():
    return distributed_cache.get_stats()
