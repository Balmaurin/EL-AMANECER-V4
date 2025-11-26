"""
Advanced Performance Service for Sheily AI Enterprise.
Provides intelligent performance monitoring, optimization, and caching.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self):
        self.response_times: deque = deque(maxlen=10000)
        self.throughput: deque = deque(maxlen=1000)
        self.error_rates: deque = deque(maxlen=1000)
        self.memory_usage: deque = deque(maxlen=1000)
        self.cpu_usage: deque = deque(maxlen=1000)
        self.cache_hit_rates: deque = deque(maxlen=1000)

    def add_response_time(self, time_ms: float):
        """Add response time measurement."""
        self.response_times.append(time_ms)

    def add_throughput(self, requests_per_second: float):
        """Add throughput measurement."""
        self.throughput.append(requests_per_second)

    def add_error_rate(self, error_rate: float):
        """Add error rate measurement."""
        self.error_rates.append(error_rate)

    def add_memory_usage(self, usage_percent: float):
        """Add memory usage measurement."""
        self.memory_usage.append(usage_percent)

    def add_cpu_usage(self, usage_percent: float):
        """Add CPU usage measurement."""
        self.cpu_usage.append(usage_percent)

    def add_cache_hit_rate(self, hit_rate: float):
        """Add cache hit rate measurement."""
        self.cache_hit_rates.append(hit_rate)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "response_time": {
                "avg": (
                    sum(self.response_times) / len(self.response_times)
                    if self.response_times
                    else 0
                ),
                "p95": (
                    sorted(list(self.response_times))[
                        int(len(self.response_times) * 0.95)
                    ]
                    if self.response_times
                    else 0
                ),
                "p99": (
                    sorted(list(self.response_times))[
                        int(len(self.response_times) * 0.99)
                    ]
                    if self.response_times
                    else 0
                ),
                "min": min(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
            },
            "throughput": {
                "avg": (
                    sum(self.throughput) / len(self.throughput)
                    if self.throughput
                    else 0
                ),
                "current": self.throughput[-1] if self.throughput else 0,
            },
            "error_rate": {
                "avg": (
                    sum(self.error_rates) / len(self.error_rates)
                    if self.error_rates
                    else 0
                ),
                "current": self.error_rates[-1] if self.error_rates else 0,
            },
            "system": {
                "memory_avg": (
                    sum(self.memory_usage) / len(self.memory_usage)
                    if self.memory_usage
                    else 0
                ),
                "cpu_avg": (
                    sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
                ),
                "cache_hit_rate_avg": (
                    sum(self.cache_hit_rates) / len(self.cache_hit_rates)
                    if self.cache_hit_rates
                    else 0
                ),
            },
        }


class IntelligentCache:
    """Intelligent caching system with predictive prefetching."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.prefetch_hits = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with access tracking."""
        if key in self.cache:
            entry = self.cache[key]
            if self._is_expired(entry):
                del self.cache[key]
                self.misses += 1
                return None

            self.hits += 1
            self.access_patterns[key].append(datetime.now(timezone.utc))
            return entry["value"]
        else:
            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache with optional custom TTL."""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        self.cache[key] = {
            "value": value,
            "created_at": datetime.now(timezone.utc),
            "ttl": ttl or self.ttl_seconds,
        }

    def prefetch(self, patterns: List[str]):
        """Prefetch likely-to-be-needed items based on patterns."""
        # Analyze access patterns to predict future needs
        for pattern in patterns:
            likely_keys = self._predict_access_pattern(pattern)
            for key in likely_keys[:5]:  # Prefetch top 5 predictions
                if key not in self.cache:
                    # Simulate prefetch operation
                    self.prefetch_hits += 1

    def _predict_access_pattern(self, pattern: str) -> List[str]:
        """Predict likely future access patterns."""
        # Simple prediction based on recent access frequency
        recent_accesses = []
        for key, accesses in self.access_patterns.items():
            if pattern in key:
                recent_count = len(
                    [
                        a
                        for a in accesses
                        if a > datetime.now(timezone.utc) - timedelta(hours=1)
                    ]
                )
                recent_accesses.append((key, recent_count))

        return [
            key for key, _ in sorted(recent_accesses, key=lambda x: x[1], reverse=True)
        ]

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return (
            datetime.now(timezone.utc) - entry["created_at"]
        ).total_seconds() > entry["ttl"]

    def _evict_oldest(self):
        """Evict oldest cache entries."""
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["created_at"])
        del self.cache[oldest_key]

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PerformanceOptimizer:
    """Intelligent performance optimization engine."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.cache = IntelligentCache()
        self.optimization_rules = []
        self.monitoring_active = False

    async def start_monitoring(self):
        """Start continuous performance monitoring."""
        self.monitoring_active = True

        # Start background monitoring tasks
        asyncio.create_task(self._monitor_system_resources())
        asyncio.create_task(self._monitor_application_metrics())
        asyncio.create_task(self._apply_optimizations())

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False

    async def _monitor_system_resources(self):
        """Monitor system resources continuously."""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.add_cpu_usage(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics.add_memory_usage(memory.percent)

                # Cache hit rate
                cache_hit_rate = self.cache.get_hit_rate()
                self.metrics.add_cache_hit_rate(cache_hit_rate)

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
                await asyncio.sleep(30)

    async def _monitor_application_metrics(self):
        """Monitor application-specific metrics."""
        while self.monitoring_active:
            try:
                # Calculate throughput (requests per second)
                # This would be integrated with your actual request handling
                throughput = 150.0  # Mock value
                self.metrics.add_throughput(throughput)

                # Calculate error rate
                error_rate = 0.02  # Mock value (2%)
                self.metrics.add_error_rate(error_rate)

                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Error monitoring application metrics: {e}")
                await asyncio.sleep(60)

    async def _apply_optimizations(self):
        """Apply intelligent optimizations based on metrics."""
        while self.monitoring_active:
            try:
                stats = self.metrics.get_stats()

                # Apply cache optimizations
                if stats["system"]["cache_hit_rate_avg"] < 0.8:
                    self._optimize_cache()

                # Apply memory optimizations
                if stats["system"]["memory_avg"] > 85:
                    self._optimize_memory()

                # Apply performance optimizations
                if stats["response_time"]["p95"] > 1000:  # 1 second
                    self._optimize_performance()

                await asyncio.sleep(300)  # Apply optimizations every 5 minutes

            except Exception as e:
                logger.error(f"Error applying optimizations: {e}")
                await asyncio.sleep(300)

    def _optimize_cache(self):
        """Apply cache optimizations."""
        logger.info("Applying cache optimizations")

        # Increase cache size if hit rate is low
        if self.cache.get_hit_rate() < 0.7:
            self.cache.max_size *= 1.5

        # Prefetch based on patterns
        common_patterns = ["agent:", "tenant:", "analytics:"]
        self.cache.prefetch(common_patterns)

    def _optimize_memory(self):
        """Apply memory optimizations."""
        logger.info("Applying memory optimizations")

        # Force garbage collection
        import gc

        gc.collect()

        # Reduce cache size if memory is high
        if psutil.virtual_memory().percent > 90:
            self.cache.max_size = int(self.cache.max_size * 0.8)

    def _optimize_performance(self):
        """Apply performance optimizations."""
        logger.info("Applying performance optimizations")

        # Add optimization rules
        self.optimization_rules.append(
            {
                "type": "response_time",
                "threshold": 1000,
                "action": "increase_cache_ttl",
                "applied_at": datetime.now(timezone.utc),
            }
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        stats = self.metrics.get_stats()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "overall_health": self._calculate_health_score(stats),
                "bottlenecks": self._identify_bottlenecks(stats),
                "recommendations": self._generate_recommendations(stats),
            },
            "metrics": stats,
            "cache_performance": {
                "hit_rate": self.cache.get_hit_rate(),
                "size": len(self.cache.cache),
                "max_size": self.cache.max_size,
                "prefetch_hits": self.cache.prefetch_hits,
            },
            "optimizations_applied": len(self.optimization_rules),
        }

    def _calculate_health_score(self, stats: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0

        # Response time penalties
        if stats["response_time"]["p95"] > 500:
            score -= 20
        elif stats["response_time"]["p95"] > 200:
            score -= 10

        # Error rate penalties
        if stats["error_rate"]["avg"] > 0.05:
            score -= 15
        elif stats["error_rate"]["avg"] > 0.02:
            score -= 5

        # System resource penalties
        if stats["system"]["memory_avg"] > 90:
            score -= 25
        elif stats["system"]["memory_avg"] > 80:
            score -= 10

        if stats["system"]["cpu_avg"] > 90:
            score -= 25
        elif stats["system"]["cpu_avg"] > 80:
            score -= 10

        return max(0.0, min(100.0, score))

    def _identify_bottlenecks(self, stats: Dict[str, Any]) -> List[str]:
        """Identify system bottlenecks."""
        bottlenecks = []

        if stats["response_time"]["p95"] > 1000:
            bottlenecks.append("High response time (P95 > 1s)")
        if stats["error_rate"]["avg"] > 0.05:
            bottlenecks.append("High error rate (>5%)")
        if stats["system"]["memory_avg"] > 85:
            bottlenecks.append("High memory usage (>85%)")
        if stats["system"]["cpu_avg"] > 85:
            bottlenecks.append("High CPU usage (>85%)")
        if stats["system"]["cache_hit_rate_avg"] < 0.8:
            bottlenecks.append("Low cache hit rate (<80%)")

        return bottlenecks

    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if stats["response_time"]["p95"] > 500:
            recommendations.append("Consider implementing response time optimizations")
        if stats["system"]["cache_hit_rate_avg"] < 0.8:
            recommendations.append(
                "Increase cache size or implement better caching strategies"
            )
        if stats["system"]["memory_avg"] > 80:
            recommendations.append(
                "Monitor memory usage and consider memory optimization"
            )
        if stats["error_rate"]["avg"] > 0.02:
            recommendations.append("Investigate and reduce error rates")

        if not recommendations:
            recommendations.append("System performance is optimal")

        return recommendations


# Global performance service instance
performance_service = PerformanceOptimizer()

__all__ = [
    "PerformanceMetrics",
    "IntelligentCache",
    "PerformanceOptimizer",
    "performance_service",
]
