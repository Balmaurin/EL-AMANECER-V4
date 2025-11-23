"""
Search cache implementation with frequency analysis and TTL.
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tools.common.errors import CacheError
from tools.monitoring.metrics import monitor_operation


@dataclass
class SearchResult:
    """Represents a cached search result"""

    query: str
    results: List[Dict]
    mode: str
    timestamp: float
    score: float
    metadata: Dict


@dataclass
class QueryStatistics:
    """Statistics for query analysis"""

    frequency: int
    last_access: float
    avg_result_count: float
    avg_score: float
    variations: List[str]


class SearchCache:
    def __init__(
        self,
        cache_dir: str,
        ttl: int = 3600,
        max_entries: int = 10000,
        min_frequency: int = 5,  # 1 hour default TTL
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.max_entries = max_entries
        self.min_frequency = min_frequency

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db_path = self.cache_dir / "search_cache.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            # Cache table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    results TEXT,
                    mode TEXT,
                    timestamp REAL,
                    score REAL,
                    metadata TEXT
                )
            """
            )

            # Query statistics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_stats (
                    query TEXT PRIMARY KEY,
                    frequency INTEGER,
                    last_access REAL,
                    avg_result_count REAL,
                    avg_score REAL,
                    variations TEXT
                )
            """
            )

            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_frequency ON query_stats(frequency)"
            )

    def get(self, query: str, mode: str) -> Optional[SearchResult]:
        """Get cached search result"""
        with monitor_operation("cache", "get_search") as span:
            try:
                # Clean expired entries
                self._clean_expired()

                query_hash = self._generate_hash(query, mode)

                with sqlite3.connect(self.db_path) as conn:
                    # Get cached result
                    cursor = conn.execute(
                        """
                        SELECT query, results, mode, timestamp, score, metadata
                        FROM cache
                        WHERE query_hash = ?
                    """,
                        (query_hash,),
                    )

                    row = cursor.fetchone()
                    if not row:
                        span.set_attribute("cache_hit", False)
                        return None

                    # Update access statistics
                    self._update_stats(
                        conn, query, len(json.loads(row[1])), float(row[4])
                    )

                    result = SearchResult(
                        query=row[0],
                        results=json.loads(row[1]),
                        mode=row[2],
                        timestamp=float(row[3]),
                        score=float(row[4]),
                        metadata=json.loads(row[5]),
                    )

                    span.set_attribute("cache_hit", True)
                    return result

            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise CacheError(f"Error retrieving from cache: {e}")

    def store(
        self,
        query: str,
        results: List[Dict],
        mode: str,
        score: float,
        metadata: Optional[Dict] = None,
    ):
        """Store search result in cache"""
        with monitor_operation("cache", "store_search") as span:
            try:
                # Ensure we have space
                self._ensure_capacity()

                query_hash = self._generate_hash(query, mode)
                timestamp = time.time()

                with sqlite3.connect(self.db_path) as conn:
                    # Store result
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache
                        (query_hash, query, results, mode, timestamp, score, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            query_hash,
                            query,
                            json.dumps(results),
                            mode,
                            timestamp,
                            score,
                            json.dumps(metadata or {}),
                        ),
                    )

                    # Update statistics
                    self._update_stats(conn, query, len(results), score)

                span.set_attribute("results", len(results))

            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise CacheError(f"Error storing in cache: {e}")

    def get_similar_queries(
        self, query: str, threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """Find similar queries based on frequency and string similarity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT query, frequency
                FROM query_stats
                WHERE frequency >= ?
                ORDER BY frequency DESC
                LIMIT 100
            """,
                (self.min_frequency,),
            )

            candidates = []
            for row in cursor:
                similarity = self._compute_similarity(query, row[0])
                if similarity >= threshold:
                    candidates.append((row[0], similarity))

            return sorted(candidates, key=lambda x: x[1], reverse=True)

    def get_popular_queries(
        self, limit: int = 10, time_window: int = 86400
    ) -> List[QueryStatistics]:  # 1 day
        """Get most popular queries within time window"""
        with sqlite3.connect(self.db_path) as conn:
            min_time = time.time() - time_window
            cursor = conn.execute(
                """
                SELECT query, frequency, last_access, avg_result_count,
                       avg_score, variations
                FROM query_stats
                WHERE last_access >= ? AND frequency >= ?
                ORDER BY frequency DESC
                LIMIT ?
            """,
                (min_time, self.min_frequency, limit),
            )

            return [
                QueryStatistics(
                    frequency=row[1],
                    last_access=row[2],
                    avg_result_count=row[3],
                    avg_score=row[4],
                    variations=json.loads(row[5]),
                )
                for row in cursor
            ]

    def _clean_expired(self):
        """Remove expired entries"""
        with sqlite3.connect(self.db_path) as conn:
            expired_time = time.time() - self.ttl
            conn.execute(
                """
                DELETE FROM cache
                WHERE timestamp < ?
            """,
                (expired_time,),
            )

    def _ensure_capacity(self):
        """Ensure cache doesn't exceed max entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]

            if count >= self.max_entries:
                # Remove 20% of oldest entries
                to_remove = count - int(self.max_entries * 0.8)
                conn.execute(
                    """
                    DELETE FROM cache
                    WHERE query_hash IN (
                        SELECT query_hash FROM cache
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                """,
                    (to_remove,),
                )

    def _update_stats(
        self, conn: sqlite3.Connection, query: str, num_results: int, score: float
    ):
        """Update query statistics"""
        timestamp = time.time()

        cursor = conn.execute(
            """
            SELECT frequency, avg_result_count, avg_score, variations
            FROM query_stats
            WHERE query = ?
        """,
            (query,),
        )

        row = cursor.fetchone()
        if row:
            # Update existing stats
            freq = row[0] + 1
            avg_results = (row[1] * row[0] + num_results) / freq
            avg_score = (row[2] * row[0] + score) / freq
            variations = json.loads(row[3])

            if query not in variations:
                variations.append(query)

            conn.execute(
                """
                UPDATE query_stats
                SET frequency = ?,
                    last_access = ?,
                    avg_result_count = ?,
                    avg_score = ?,
                    variations = ?
                WHERE query = ?
            """,
                (
                    freq,
                    timestamp,
                    avg_results,
                    avg_score,
                    json.dumps(variations),
                    query,
                ),
            )
        else:
            # Insert new stats
            conn.execute(
                """
                INSERT INTO query_stats
                (query, frequency, last_access, avg_result_count,
                 avg_score, variations)
                VALUES (?, 1, ?, ?, ?, ?)
            """,
                (query, timestamp, float(num_results), score, json.dumps([query])),
            )

    def _generate_hash(self, query: str, mode: str) -> str:
        """Generate unique hash for query+mode combination"""
        import hashlib

        return hashlib.sha256(f"{query}:{mode}".encode("utf-8")).hexdigest()

    def _compute_similarity(self, query1: str, query2: str) -> float:
        """Compute string similarity between queries"""
        from difflib import SequenceMatcher

        return SequenceMatcher(None, query1.lower(), query2.lower()).ratio()
