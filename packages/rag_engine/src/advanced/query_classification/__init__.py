"""
Query Classification System for RAG
Based on EMNLP 2024 Paper Section A.1

Classifies queries as "retrieval required" vs "no retrieval required"
using BERT-base-multilingual-cased with 95% accuracy target.
"""

from .classifier import QueryClassificationResult, QueryClassifier

__all__ = ["QueryClassifier", "QueryClassificationResult"]
