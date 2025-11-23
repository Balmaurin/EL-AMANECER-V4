"""
Document retrieval module for RAG systems.

This module provides comprehensive search functionality across multiple
retrieval strategies:

Lexical Search:
- BM25 ranking with Whoosh backend
- Multi-field search
- Configurable field weights

Dense Vector Search:
- HNSW approximate nearest neighbor search
- Multiple vector store backends (local, Qdrant, Milvus)
- Optimized vector similarity calculations

Hybrid Search:
- Score fusion algorithms
- Configurable strategy weights
- Result reranking
- Quality-based filtering

The module supports efficient search across large document collections
with configurable trade-offs between speed and accuracy.
"""

__all__ = ["unified_search", "search_bm25", "search_local_hnsw", "dense_search"]


# Lazy attribute loading to avoid heavy imports and side-effects at package import time
def __getattr__(name):  # pragma: no cover
    if name == "unified_search":
        from .search_unified import unified_search

        return unified_search
    if name == "search_bm25":
        from .search_bm25_whoosh import search_bm25

        return search_bm25
    if name == "search_local_hnsw":
        from .search_local_hnsw import search_local_hnsw

        return search_local_hnsw
    if name == "dense_search":
        from .dense_switch import dense_search

        return dense_search
    raise AttributeError(name)
