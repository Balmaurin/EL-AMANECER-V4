"""
Summarization Methods for RAG Context Compression
Based on EMNLP 2024 Paper Section A.5

Implements:
- Selective Context (self-information based)
- LongLLMLingua (compression with retention)
- Recomp (extractive + abstractive)
"""

from .context_summarizer import (
    ContextSummarizer,
    SummarizationMethod,
    SummarizationResult,
)

__all__ = ["ContextSummarizer", "SummarizationResult", "SummarizationMethod"]
