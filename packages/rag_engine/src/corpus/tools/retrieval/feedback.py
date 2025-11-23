"""
Search feedback and result diversification system.
Implements click feedback, result diversification, and dynamic scoring.
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = None

from tools.common.errors import FeedbackError
from tools.monitoring.metrics import monitor_operation


@dataclass
class Click:
    """Click feedback data"""

    query: str
    result_id: str
    position: int
    timestamp: float
    dwell_time: Optional[float] = None
    is_last_click: bool = False


@dataclass
class ResultScore:
    """Dynamic result scoring"""

    base_score: float
    click_score: float
    diversity_score: float
    final_score: float


class FeedbackManager:
    def __init__(
        self,
        db_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        click_window: int = 86400,  # 1 day
        min_clicks: int = 5,
    ):
        self.db_path = db_path
        self.click_window = click_window
        self.min_clicks = min_clicks
        self.embedding_model = embedding_model
        self._embedder = None

        # Initialize database
        self._init_database()

    @property
    def embedder(self):
        """Lazy load embedder to avoid torch import at module level"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            # Click feedback table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clicks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    result_id TEXT,
                    position INTEGER,
                    timestamp REAL,
                    dwell_time REAL,
                    is_last_click BOOLEAN
                )
            """
            )

            # Result scores table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS result_scores (
                    query_result TEXT PRIMARY KEY,
                    click_count INTEGER,
                    avg_position REAL,
                    avg_dwell_time REAL,
                    last_click_ratio REAL,
                    score REAL,
                    last_update REAL
                )
            """
            )

            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON clicks(query)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_click_time ON clicks(timestamp)"
            )

    def record_click(self, click: Click):
        """Record click feedback"""
        with monitor_operation("feedback", "record_click") as span:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO clicks
                        (query, result_id, position, timestamp,
                         dwell_time, is_last_click)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            click.query,
                            click.result_id,
                            click.position,
                            click.timestamp,
                            click.dwell_time,
                            click.is_last_click,
                        ),
                    )

                    # Update result scores
                    self._update_result_score(conn, click)

                span.set_attribute("query", click.query)
                span.set_attribute("position", click.position)

            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise FeedbackError(f"Error recording click: {e}")

    def rerank_results(
        self,
        query: str,
        results: List[Dict],
        alpha: float = 0.3,  # Weight for click feedback
        beta: float = 0.3,  # Weight for diversity
    ) -> List[Dict]:
        """Rerank results using click feedback and diversity"""
        with monitor_operation("feedback", "rerank") as span:
            try:
                if not results:
                    return results

                # Get result embeddings
                texts = [r.get("text", "") for r in results]
                embeddings = self.embedder.encode(texts)

                # Initialize scores
                scores = []
                selected_embeddings = []
                final_results = []

                # Score and select results iteratively
                remaining = list(enumerate(results))

                while remaining:
                    curr_scores = []

                    for idx, result in remaining:
                        # Get base relevance score
                        base_score = result.get("score", 0.5)

                        # Get click feedback score
                        click_score = self._get_click_score(query, result)

                        # Compute diversity score
                        if selected_embeddings:
                            div_score = self._compute_diversity_score(
                                embeddings[idx], selected_embeddings
                            )
                        else:
                            div_score = 1.0

                        # Compute final score
                        final_score = (
                            base_score * (1 - alpha - beta)
                            + click_score * alpha
                            + div_score * beta
                        )

                        curr_scores.append(
                            (
                                idx,
                                ResultScore(
                                    base_score=base_score,
                                    click_score=click_score,
                                    diversity_score=div_score,
                                    final_score=final_score,
                                ),
                            )
                        )

                    # Select result with highest score
                    if curr_scores:
                        selected_idx, score = max(
                            curr_scores, key=lambda x: x[1].final_score
                        )

                        # Add to final results
                        result = results[selected_idx].copy()
                        result["feedback_score"] = score.final_score
                        final_results.append(result)

                        # Update selected embeddings
                        selected_embeddings.append(embeddings[selected_idx])

                        # Remove selected result
                        remaining = [(i, r) for i, r in remaining if i != selected_idx]

                span.set_attribute("results", len(final_results))
                return final_results

            except Exception as e:
                span.set_status("error")
                span.set_attribute("error", str(e))
                raise FeedbackError(f"Error reranking results: {e}")

    def _update_result_score(self, conn: sqlite3.Connection, click: Click):
        """Update result score based on click feedback"""
        # Get existing score data
        cursor = conn.execute(
            """
            SELECT click_count, avg_position, avg_dwell_time,
                   last_click_ratio, score
            FROM result_scores
            WHERE query_result = ?
        """,
            (f"{click.query}:{click.result_id}",),
        )

        row = cursor.fetchone()

        if row:
            # Update existing score
            count = row[0] + 1
            avg_pos = (row[1] * row[0] + click.position) / count
            avg_dwell = (row[2] * row[0] + (click.dwell_time or 0)) / count
            last_ratio = (row[3] * row[0] + int(click.is_last_click)) / count

            # Compute new score
            new_score = self._compute_click_score(count, avg_pos, avg_dwell, last_ratio)

            conn.execute(
                """
                UPDATE result_scores
                SET click_count = ?,
                    avg_position = ?,
                    avg_dwell_time = ?,
                    last_click_ratio = ?,
                    score = ?,
                    last_update = ?
                WHERE query_result = ?
            """,
                (
                    count,
                    avg_pos,
                    avg_dwell,
                    last_ratio,
                    new_score,
                    time.time(),
                    f"{click.query}:{click.result_id}",
                ),
            )
        else:
            # Insert new score
            score = self._compute_click_score(
                1, click.position, click.dwell_time or 0, int(click.is_last_click)
            )

            conn.execute(
                """
                INSERT INTO result_scores
                (query_result, click_count, avg_position,
                 avg_dwell_time, last_click_ratio, score, last_update)
                VALUES (?, 1, ?, ?, ?, ?, ?)
            """,
                (
                    f"{click.query}:{click.result_id}",
                    click.position,
                    click.dwell_time or 0,
                    int(click.is_last_click),
                    score,
                    time.time(),
                ),
            )

    def _get_click_score(self, query: str, result: Dict) -> float:
        """Get click-based score for a result"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT score
                FROM result_scores
                WHERE query_result = ?
                  AND last_update >= ?
                  AND click_count >= ?
            """,
                (
                    f"{query}:{result.get('id', '')}",
                    time.time() - self.click_window,
                    self.min_clicks,
                ),
            )

            row = cursor.fetchone()
            return row[0] if row else 0.5

    def _compute_click_score(
        self,
        count: int,
        avg_position: float,
        avg_dwell_time: float,
        last_click_ratio: float,
    ) -> float:
        """Compute score from click statistics"""
        # Position score (higher positions are better)
        pos_score = 1.0 / (1.0 + avg_position)

        # Dwell time score (longer is better, with diminishing returns)
        dwell_score = min(avg_dwell_time / 30.0, 1.0)  # Cap at 30 seconds

        # Last click bonus
        last_bonus = last_click_ratio * 0.2  # Up to 20% bonus

        # Combine scores
        score = pos_score * 0.4 + dwell_score * 0.4 + last_bonus

        return score

    def _compute_diversity_score(
        self, embedding: np.ndarray, selected_embeddings: List[np.ndarray]
    ) -> float:
        """Compute diversity score against selected results"""
        if not selected_embeddings:
            return 1.0

        # Compute similarities
        sims = cosine_similarity([embedding], selected_embeddings)[0]

        # Convert to diversity score (lower similarity = higher diversity)
        return 1.0 - np.max(sims)

    def get_click_stats(
        self, query: Optional[str] = None, time_window: int = 86400
    ) -> Dict:
        """Get click statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Base query
            base_query = """
                SELECT COUNT(*) as total_clicks,
                       AVG(position) as avg_position,
                       AVG(dwell_time) as avg_dwell_time,
                       SUM(CASE WHEN is_last_click THEN 1 ELSE 0 END) as last_clicks
                FROM clicks
                WHERE timestamp >= ?
            """

            params = [time.time() - time_window]

            if query:
                base_query += " AND query = ?"
                params.append(query)

            cursor = conn.execute(base_query, params)
            row = cursor.fetchone()

            stats.update(
                {
                    "total_clicks": row[0],
                    "avg_position": row[1],
                    "avg_dwell_time": row[2],
                    "last_clicks": row[3],
                }
            )

            return stats
