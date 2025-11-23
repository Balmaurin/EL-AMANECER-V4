"""
RAG system evaluation module.

This module provides tools for evaluating retrieval-augmented generation (RAG)
system performance using various metrics:

Retrieval Metrics:
- Precision @ K
- Recall @ K
- Mean Reciprocal Rank (MRR)
- Coverage of gold standard snippets

Quality Metrics:
- Context relevance
- Answer accuracy
- Response completeness
- Query specificity

The module supports evaluation against gold standard datasets and provides
detailed performance analysis.
"""

from .eval_rag import EvalQuery, EvalResults, calculate_mrr, evaluate_retrieval

__all__ = ["EvalQuery", "EvalResults", "evaluate_retrieval", "calculate_mrr"]
