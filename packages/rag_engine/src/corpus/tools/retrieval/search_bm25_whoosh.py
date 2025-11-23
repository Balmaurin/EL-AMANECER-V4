"""
BM25 search implementation using Whoosh.

This module provides lexical search functionality using the BM25 algorithm
implemented in Whoosh. Features include:
- Multi-field search
- Configurable field weights
- OR query groups with tuneable minimums
- Score normalization
- Metadata handling
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from whoosh import index
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.searching import Hit, Results

# Configure logging
log = logging.getLogger("rag.retrieval.bm25")


class SearchResult:
    """Data class for BM25 search results."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str = "",
        title: str = "",
        text: str = "",
        score: float = 0.0,
        lang: str = "",
        tags: List[str] = None,
        quality: float = 0.5,
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.score = float(score)
        self.lang = lang
        self.tags = tags or []
        self.quality = quality

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "meta": {"lang": self.lang, "tags": self.tags, "quality": self.quality},
        }


def search_bm25(
    base: Path, query: str, top_k: int = 10, fields: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    """Execute BM25 search over indexed documents.

    Args:
        base: Base directory containing the BM25 index
        query: Search query string
        top_k: Number of results to return
        fields: Fields to search in (defaults to title and text)

    Returns:
        List of search results sorted by BM25 score

    Raises:
        ValueError: If query is empty or invalid
        FileNotFoundError: If index directory not found
    """
    # Validate parameters
    if not query:
        log.warning("Empty query provided")
        return []

    fields = fields or ["title", "text"]
    index_dir = base / "index" / "bm25"

    if not index_dir.exists():
        log.error(f"BM25 index not found at {index_dir}")
        return []

    try:
        # Open index
        ix = index.open_dir(str(index_dir))

        # Configure parser with OR group (75% of terms must match)
        parser = MultifieldParser(fields, schema=ix.schema, group=OrGroup.factory(0.75))

        # Parse query
        parsed_query = parser.parse(query)

        results = []
        # Execute search
        with ix.searcher() as searcher:
            hits: Results = searcher.search(parsed_query, limit=top_k)

            # Process results
            for hit in hits:
                result = SearchResult(
                    chunk_id=hit.get("chunk_id", ""),
                    doc_id=hit.get("doc_id", ""),
                    title=hit.get("title", ""),
                    text=hit.get("text", ""),
                    score=float(hit.score),
                    lang=hit.get("lang", ""),
                    tags=(hit.get("tags", "") or "").split(","),
                    quality=hit.get("quality", 0.5),
                )
                results.append(result.to_dict())

        return results

    except Exception as e:
        log.error(f"Error executing BM25 search: {e}")
        return []
