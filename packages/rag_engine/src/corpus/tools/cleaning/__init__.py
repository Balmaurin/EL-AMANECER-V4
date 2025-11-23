"""
Text cleaning and quality assessment module for RAG systems.

This module provides comprehensive text preprocessing capabilities:

Quality Assessment:
- Information entropy calculation
- Stopword distribution analysis
- Character and digit ratio analysis
- Overall quality scoring

Text Cleaning:
- HTML removal
- PII (Personally Identifiable Information) redaction
- Unicode normalization
- Whitespace normalization
- Language detection
- Deduplication via LSH
"""

from .normalize import normalize_corpus
from .quality import digit_ratio, entropy, quality_score, stopword_ratio

__all__ = [
    "normalize_corpus",
    "quality_score",
    "entropy",
    "stopword_ratio",
    "digit_ratio",
]
