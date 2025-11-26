"""
Text embedding module for RAG systems.

This module provides functionality for converting text into dense vector
embeddings using transformer models. Key features include:

Embedding Generation:
- Efficient batched processing
- Multiple model support (sentence-transformers)
- GPU acceleration with automatic device selection
- Progress tracking with ETA estimation

Caching System:
- SQLite-based persistence with JSON serialization
- Thread-safe operations for concurrent access
- Automatic cache management and cleanup
- Error recovery with retry mechanisms

The module supports various embedding models and includes optimizations
for both memory usage and processing speed.

Example:
    >>> from tools.embedding import embed_corpus, EmbCache
    >>> embed_corpus("main", Path("data/corpus"))
    >>> cache = EmbCache()
    >>> embeddings = cache.get_embeddings(["text1", "text2"])

Returns:
    embed_corpus: None
        Generates and caches embeddings for all chunks
    EmbCache: EmbCache
        Cache manager for storing and retrieving embeddings
"""

from .embed import embed_corpus
from .embed_cache import EmbCache

__all__ = ["embed_corpus", "EmbCache"]
