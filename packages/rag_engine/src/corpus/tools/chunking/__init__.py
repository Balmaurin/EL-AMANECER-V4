"""
Text chunking module for RAG document processing.

This module provides tools for splitting documents into semantically meaningful chunks
while preserving context and maintaining chunk size constraints. It includes:

- Semantic chunking with configurable overlap
- Sentence boundary detection
- Quality-aware chunk generation
- Support for multiple languages
"""

from .semantic_split import semantic_chunks, sentence_split

__all__ = ["semantic_chunks", "sentence_split"]
