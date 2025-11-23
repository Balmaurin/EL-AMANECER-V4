"""
Advanced RAG Techniques Implementation for Sheily AI
Based on EMNLP 2024 Paper: "A Survey on Retrieval-Augmented Generation"

This module implements all advanced RAG techniques from the paper:
- Query Classification
- Advanced Chunking
- Multiple Retrieval Methods (HyDE, Query Rewriting, etc.)
- Reranking Systems
- Summarization Methods
- Generator Fine-tuning
- Comprehensive Evaluation with RAGAs
"""

__version__ = "1.0.0"
__author__ = "Sheily AI Team"

from .chunking import AdvancedChunker, ChunkingMethod, ChunkingResult
from .evaluation import EvaluationMetric, EvaluationResult, RAGEvaluator
from .integration import IntegratedRAGResult, RAGIntegrator
from .parametric_rag import ParametricDocument, ParametricRAG, ParametricRAGResult
from .query_classification import QueryClassificationResult, QueryClassifier
from .reranking import Reranker, RerankerType, RerankingResult
from .retrieval import AdvancedRetriever, RetrievalMethod, RetrievalResult
from .summarization import ContextSummarizer, SummarizationMethod, SummarizationResult

__all__ = [
    # Core Components
    "QueryClassifier",
    "QueryClassificationResult",
    "AdvancedChunker",
    "ChunkingResult",
    "ChunkingMethod",
    "AdvancedRetriever",
    "RetrievalResult",
    "RetrievalMethod",
    "Reranker",
    "RerankingResult",
    "RerankerType",
    "ContextSummarizer",
    "SummarizationResult",
    "SummarizationMethod",
    "RAGEvaluator",
    "EvaluationResult",
    "EvaluationMetric",
    # Parametric RAG (New Paradigm)
    "ParametricRAG",
    "ParametricDocument",
    "ParametricRAGResult",
    # Integration
    "RAGIntegrator",
    "IntegratedRAGResult",
]
