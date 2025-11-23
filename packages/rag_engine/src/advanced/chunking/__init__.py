"""
Advanced Chunking Techniques for RAG
Based on EMNLP 2024 Paper Section A.2

Implements:
- Small-to-big chunking (175/512 tokens)
- Sliding window with 20-token overlap
- Comparison of chunking strategies
"""

from .advanced_chunker import AdvancedChunker, ChunkingMethod, ChunkingResult

__all__ = ["AdvancedChunker", "ChunkingResult", "ChunkingMethod"]
