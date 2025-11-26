'''
Sheily Core Cache Module
'''

from .smart_cache import (
    SmartCache,
    SemanticCache,
    ResponseCache,
    CacheEntry,
    get_smart_cache,
    cached_query,
    get_cache_stats
)

# Alias for backward compatibility
cached = cached_query

__all__ = [
    'SmartCache',
    'SemanticCache', 
    'ResponseCache',
    'CacheEntry',
    'get_smart_cache',
    'cached_query',
    'cached',  # Alias
    'get_cache_stats'
]
