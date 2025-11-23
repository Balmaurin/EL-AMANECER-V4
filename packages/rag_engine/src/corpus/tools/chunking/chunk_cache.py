"""
Caching system for frequently accessed chunks.
Implements LRU cache with disk persistence.
"""

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tools.common.errors import CacheError
from tools.monitoring.metrics import monitor_operation


@dataclass
class ChunkMetadata:
    """Metadata for cached chunks"""

    chunk_hash: str
    source_file: str
    creation_time: str
    last_access: str
    access_count: int
    chunk_size: int
    quality_score: float


class ChunkCache:
    def __init__(
        self,
        cache_dir: str,
        max_cache_size: int = 1_000_000_000,
        vacuum_threshold: float = 0.8,
    ):  # 1GB
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size
        self.vacuum_threshold = vacuum_threshold
        self.lock = threading.Lock()

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database
        self.db_path = self.cache_dir / "chunk_metadata.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for chunk metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_hash TEXT PRIMARY KEY,
                    source_file TEXT,
                    creation_time TEXT,
                    last_access TEXT,
                    access_count INTEGER,
                    chunk_size INTEGER,
                    quality_score REAL
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_access
                ON chunks(last_access)
            """
            )

    def get_chunk(self, chunk_hash: str) -> Optional[str]:
        """Retrieve a chunk from cache"""
        with monitor_operation("cache", "get_chunk") as span:
            try:
                chunk_path = self.cache_dir / f"{chunk_hash}.txt"

                if not chunk_path.exists():
                    span.set_attribute("cache_hit", False)
                    return None

                with self.lock:
                    # Update access metadata
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            """
                            UPDATE chunks
                            SET last_access = ?, access_count = access_count + 1
                            WHERE chunk_hash = ?
                        """,
                            (datetime.utcnow().isoformat(), chunk_hash),
                        )

                    # Read chunk content
                    chunk_content = chunk_path.read_text(encoding="utf-8")

                span.set_attribute("cache_hit", True)
                return chunk_content

            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise CacheError(f"Error retrieving chunk: {e}")

    def store_chunk(self, chunk: str, source_file: str, quality_score: float) -> str:
        """Store a new chunk in cache"""
        with monitor_operation("cache", "store_chunk") as span:
            try:
                # Generate chunk hash
                chunk_hash = self._generate_hash(chunk)
                chunk_size = len(chunk.encode("utf-8"))

                # Check cache size and vacuum if needed
                if self._get_cache_size() > self.max_cache_size * self.vacuum_threshold:
                    self._vacuum_cache()

                with self.lock:
                    # Store chunk content
                    chunk_path = self.cache_dir / f"{chunk_hash}.txt"
                    chunk_path.write_text(chunk, encoding="utf-8")

                    # Store metadata
                    now = datetime.utcnow().isoformat()
                    metadata = ChunkMetadata(
                        chunk_hash=chunk_hash,
                        source_file=source_file,
                        creation_time=now,
                        last_access=now,
                        access_count=0,
                        chunk_size=chunk_size,
                        quality_score=quality_score,
                    )

                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO chunks
                            (chunk_hash, source_file, creation_time, last_access,
                             access_count, chunk_size, quality_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                metadata.chunk_hash,
                                metadata.source_file,
                                metadata.creation_time,
                                metadata.last_access,
                                metadata.access_count,
                                metadata.chunk_size,
                                metadata.quality_score,
                            ),
                        )

                span.set_attribute("chunk_size", chunk_size)
                return chunk_hash

            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise CacheError(f"Error storing chunk: {e}")

    def get_metadata(self, chunk_hash: str) -> Optional[ChunkMetadata]:
        """Get metadata for a cached chunk"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM chunks WHERE chunk_hash = ?
            """,
                (chunk_hash,),
            )
            row = cursor.fetchone()

        if row:
            return ChunkMetadata(
                chunk_hash=row[0],
                source_file=row[1],
                creation_time=row[2],
                last_access=row[3],
                access_count=row[4],
                chunk_size=row[5],
                quality_score=row[6],
            )
        return None

    def _generate_hash(self, chunk: str) -> str:
        """Generate stable hash for chunk content"""
        return hashlib.sha256(chunk.encode("utf-8")).hexdigest()

    def _get_cache_size(self) -> int:
        """Get total size of cached chunks"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT SUM(chunk_size) FROM chunks")
            return cursor.fetchone()[0] or 0

    def _vacuum_cache(self):
        """Remove least recently accessed chunks until cache is under threshold"""
        with monitor_operation("cache", "vacuum") as span:
            try:
                target_size = self.max_cache_size * 0.7  # Reduce to 70% capacity

                with self.lock:
                    with sqlite3.connect(self.db_path) as conn:
                        # Get chunks to remove
                        cursor = conn.execute(
                            """
                            SELECT chunk_hash, chunk_size
                            FROM chunks
                            ORDER BY last_access ASC
                        """
                        )

                        current_size = self._get_cache_size()
                        chunks_to_remove = []

                        for chunk_hash, size in cursor:
                            if current_size <= target_size:
                                break
                            chunks_to_remove.append(chunk_hash)
                            current_size -= size

                        # Remove chunks
                        for chunk_hash in chunks_to_remove:
                            chunk_path = self.cache_dir / f"{chunk_hash}.txt"
                            if chunk_path.exists():
                                chunk_path.unlink()

                        # Remove metadata
                        if chunks_to_remove:
                            placeholders = ",".join("?" * len(chunks_to_remove))
                            query = (
                                "DELETE FROM chunks WHERE chunk_hash IN ({})".format(
                                    placeholders
                                )
                            )  # nosec B608
                            conn.execute(query, chunks_to_remove)

                span.set_attribute("chunks_removed", len(chunks_to_remove))

            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise CacheError(f"Error vacuuming cache: {e}")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Total chunks and size
            cursor = conn.execute(
                """
                SELECT COUNT(*), SUM(chunk_size), SUM(access_count)
                FROM chunks
            """
            )
            row = cursor.fetchone()
            stats["total_chunks"] = row[0] or 0
            stats["total_size"] = row[1] or 0
            stats["total_accesses"] = row[2] or 0

            # Most accessed chunks
            cursor = conn.execute(
                """
                SELECT chunk_hash, access_count
                FROM chunks
                ORDER BY access_count DESC
                LIMIT 10
            """
            )
            stats["most_accessed"] = [
                {"chunk_hash": row[0], "access_count": row[1]} for row in cursor
            ]

            return stats
