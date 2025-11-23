"""
Unified search implementation combining multiple retrieval strategies.

This module implements a hybrid search approach that combines:
- BM25 lexical search
- Dense vector similarity search
- Query expansion and rewriting
- Click feedback and diversity
- Caching and optimization
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from tools.index.index_bm25_whoosh import WhooshSearcher
from tools.retrieval.feedback import FeedbackManager
from tools.retrieval.query_expansion import QueryExpander
from tools.retrieval.search_cache import SearchCache

# Configure logging
log = logging.getLogger("rag.search")


class UnifiedSearcher:
    """Advanced unified search with caching, expansion and feedback"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        base_path: Optional[str] = None,
    ):
        # Load configuration
        if config_path:
            with open(config_path) as f:
                self.config: Dict[str, Any] = yaml.safe_load(f) or {}
        else:
            self.config: Dict[str, Any] = {}

        # Initialize components
        self.cache = SearchCache(
            cache_dir=cache_dir or "cache/search",
            ttl=int(self.config.get("cache_ttl", 3600)),
            max_entries=int(self.config.get("max_cache_entries", 10000)),
        )

        self.expander = QueryExpander()

        self.feedback = FeedbackManager(
            db_path=str(Path(cache_dir or "cache/search") / "feedback.db")
        )

        # Initialize searchers
        default_base = base_path or "data/indices"
        self.bm25_searcher = WhooshSearcher(base_path=Path(default_base))

    async def search(
        self,
        query: str,
        mode: str = "bm25",
        top_k: int = 10,
        expand_query: bool = True,
        use_feedback: bool = True,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute search and return up to top_k results.
        """
        if not query:
            return []

        # Try cache first
        cached = self.cache.get(query, mode)
        if cached:
            return cached.results[:top_k]

        results: List[Dict[str, Any]] = []

        # Currently support BM25 via WhooshSearcher
        try:
            if hasattr(self.bm25_searcher, "search"):
                maybe_results = self.bm25_searcher.search(query, limit=top_k)
                # Handle both async and sync searchers
                if hasattr(maybe_results, "__await__"):
                    results = await maybe_results  # type: ignore
                else:
                    results = list(maybe_results or [])
        except Exception:
            results = []

        # Cache results with a simple average score
        try:
            scores = [float(r.get("score", 0)) for r in results]
            avg_score = float(sum(scores) / len(scores)) if scores else 0.0
            self.cache.store(
                query=query,
                results=results[:top_k],
                mode=mode,
                score=avg_score,
                metadata={"expansions": []},
            )
        except Exception:
            pass

        return results[:top_k]

    def record_click(
        self,
        query: str,
        result_id: str,
        position: int,
        dwell_time: Optional[float] = None,
        is_last_click: bool = False,
    ):
        """Record click feedback"""
        from tools.retrieval.feedback import Click

        click = Click(
            query=query,
            result_id=result_id,
            position=position,
            timestamp=time.time(),
            dwell_time=dwell_time,
            is_last_click=is_last_click,
        )

        self.feedback.record_click(click)

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        cache_stats = getattr(self.cache, "get_stats", lambda: {})()
        click_stats = (
            self.feedback.get_click_stats()
            if hasattr(self.feedback, "get_click_stats")
            else {}
        )
        return {"cache_stats": cache_stats, "click_stats": click_stats}


import os
import time
from datetime import datetime, timezone
from typing import cast

import numpy as np

from tools.retrieval.bm25_switch import lexical_search
from tools.retrieval.dense_switch import dense_search
from tools.retrieval.rerank import build_reranker

# Configure logging
log = logging.getLogger("rag.retrieval.unified")


class SearchResult:
    """Data class for search results."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str = "",
        title: str = "",
        text: str = "",
        score: float = 0.0,
        source: str = "",
        quality: float = 0.5,
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.score = float(score)
        self.source = source
        self.quality = quality

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "quality": self.quality,
        }


def _load_config() -> Dict[str, Any]:
    """Load search configuration from universal config file.

    Returns:
        Dictionary containing search configuration

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file is invalid
    """
    try:
        with open("config/universal.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        log.error("Configuration file not found")
        raise
    except yaml.YAMLError as e:
        log.error(f"Error parsing configuration: {e}")
        raise


def unified_search(
    branch: str, base: Path, query: str, top_k: int = 6, mode: str = "hybrid"
) -> List[Dict[str, Any]]:
    """Execute unified search across multiple retrieval strategies.

    Args:
        branch: Source branch or version identifier
        base: Base directory containing indices
        query: Search query string
        top_k: Number of results to return
        mode: Search mode ('hybrid', 'bm25', 'dense', or 'expanded')

    Returns:
        List of search results sorted by score

    Raises:
        ValueError: If search mode is invalid
        FileNotFoundError: If required indices not found
    """
    # Validate parameters
    if not query:
        log.warning("Empty query provided")
        return []

    if mode not in ["hybrid", "bm25", "dense", "expanded"]:
        raise ValueError(f"Invalid search mode: {mode}")

    # Load configuration
    cfg = _load_config()
    retrieval_cfg = cfg.get("retrieval", {})

    # Execute BM25 search
    search_fields = retrieval_cfg.get("lexical", {}).get("fields", ["title", "text"])

    try:
        bm25_hits = lexical_search(
            base=base, query=query, top_k=top_k, fields=search_fields
        )
        # Ensure source label present
        for h in bm25_hits:
            h.setdefault("source", "bm25")
    except Exception as e:
        log.error(f"BM25 search failed: {e}")
        bm25_hits = []

    # Return BM25 results if that's all we need
    if mode == "bm25":
        return bm25_hits

    # Execute dense search
    try:
        dense_hits = dense_search(base=base, query=query, top_k=top_k)
        for h in dense_hits:
            h.setdefault("source", "dense")
    except Exception as e:
        log.error(f"Dense search failed: {e}")
        dense_hits = []

    # Handle hybrid search mode
    if mode == "hybrid":
        # Create rank dictionaries
        dense_ranks = {hit["chunk_id"]: i for i, hit in enumerate(dense_hits)}
        bm25_ranks = {hit["chunk_id"]: i for i, hit in enumerate(bm25_hits)}

        # Get unique document IDs
        unique_ids = list({*dense_ranks.keys(), *bm25_ranks.keys()})

        # Configure fusion parameters
        K = 8.0  # Rank smoothing factor
        weights = {"dense": 0.6, "bm25": 0.4}

        # Fuse results
        combined_results = []

        for chunk_id in unique_ids:
            # Find hits from both sources
            dense_hit = next((h for h in dense_hits if h["chunk_id"] == chunk_id), None)
            bm25_hit = next((h for h in bm25_hits if h["chunk_id"] == chunk_id), None)

            # Calculate scores
            dense_score = (1.0 / (K + dense_ranks.get(chunk_id, 9999))) * weights[
                "dense"
            ]
            bm25_score = (1.0 / (K + bm25_ranks.get(chunk_id, 9999))) * weights["bm25"]

            # Create combined result
            base_hit = dense_hit or bm25_hit
            result = SearchResult(
                chunk_id=chunk_id,
                doc_id=base_hit.get("doc_id", ""),
                title=base_hit.get("title", ""),
                text=base_hit.get("text", ""),
                score=float(dense_score + bm25_score),
                source="+".join(
                    s for s, flag in [("dense", dense_hit), ("bm25", bm25_hit)] if flag
                ),
                quality=base_hit.get("quality", 0.5),
            )
            combined_results.append(result.to_dict())

        # Sort fused list and prepare candidates
        combined_results.sort(key=lambda x: -x["score"])
        # Determine how many candidates to pass to reranker
        rr_cfg = retrieval_cfg.get("rerank", {})
        try:
            env_top_n = int(os.getenv("RAG_retrieval__rerank__top_n", "0") or "0")
        except Exception:
            env_top_n = 0
        cfg_top_n = int(rr_cfg.get("top_n", 0) or 0)
        top_n = env_top_n or cfg_top_n or max(top_k * 3, 30)
        candidates = combined_results[: min(top_n, len(combined_results))]
        results: List[Dict[str, Any]] = candidates[:top_k]
        # Optional rerank stage
        try:
            if rr_cfg.get("enabled", True):
                # Allow env override for device hint
                device = os.getenv(
                    "RAG_retrieval__rerank__device_hint",
                    rr_cfg.get("device_hint", "cpu"),
                )
                model = rr_cfg.get("model", "BAAI/bge-reranker-base")
                reranker = build_reranker(enabled=True, model=model, device_hint=device)  # type: ignore
                results = reranker.rerank(query, candidates, top_k)
                # annotate reranker info for diagnostics
                try:
                    name = getattr(reranker, "name", "unknown")
                    for h in results:
                        meta = h.get("meta") or {}
                        meta["reranked_by"] = name
                        h["meta"] = meta
                except Exception:
                    pass
        except Exception as e:
            logging.debug(f"Reranker failed in hybrid search: {e}")
        return results

    # Default to dense results with BM25 fallback
    results: List[Dict[str, Any]] = cast(
        List[Dict[str, Any]], dense_hits if dense_hits else bm25_hits
    )
    # Optional rerank for non-hybrid paths as well
    try:
        rr_cfg = retrieval_cfg.get("rerank", {})
        if rr_cfg.get("enabled", True):
            device = os.getenv(
                "RAG_retrieval__rerank__device_hint", rr_cfg.get("device_hint", "cpu")
            )
            model = rr_cfg.get("model", "BAAI/bge-reranker-base")
            reranker = build_reranker(enabled=True, model=model, device_hint=device)  # type: ignore
            # For non-hybrid paths, we already requested top_k from backend
            # Respect optional top_n override only if we have more than top_k
            try:
                env_top_n = int(os.getenv("RAG_retrieval__rerank__top_n", "0") or "0")
            except Exception:
                env_top_n = 0
            cfg_top_n = int(rr_cfg.get("top_n", 0) or 0)
            top_n = env_top_n or cfg_top_n or top_k
            candidates2 = results[: min(top_n, len(results))]
            results = reranker.rerank(query, candidates2, top_k)
            try:
                name = getattr(reranker, "name", "unknown")
                for h in results:
                    meta = h.get("meta") or {}
                    meta["reranked_by"] = name
                    h["meta"] = meta
            except Exception:
                pass
    except Exception as e:
        logging.debug(f"Reranker failed in default search: {e}")
    return results
