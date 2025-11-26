"""
Advanced Retrieval Methods for RAG
Based on EMNLP 2024 Paper Section A.3

Implements:
- HyDE (Hypothetical Document Embedding)
- Query Rewriting with Zephyr-7b-alpha
- Query Decomposition with GPT-3.5-turbo
- Hybrid Search optimization (α=0.3)
"""

from .advanced_retriever import AdvancedRetriever, RetrievalMethod, RetrievalResult

__all__ = ["AdvancedRetriever", "RetrievalResult", "RetrievalMethod"]
