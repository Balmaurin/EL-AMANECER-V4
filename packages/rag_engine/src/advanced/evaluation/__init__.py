"""
RAG Evaluation with RAGAs
Based on EMNLP 2024 Paper Section A.7

Implements comprehensive RAG evaluation:
- Faithfulness: Answer consistency with context
- Context Relevancy: Context relevance to query
- Answer Relevancy: Answer relevance to query
- Answer Correctness: Answer accuracy vs ground truth
"""

from .rag_evaluator import EvaluationMetric, EvaluationResult, RAGEvaluator

__all__ = ["RAGEvaluator", "EvaluationResult", "EvaluationMetric"]
