#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Evaluation with RAGAs Framework
Based on EMNLP 2024 Paper Section A.7

Implements comprehensive RAG evaluation metrics:
- Faithfulness: Answer consistency with retrieved context
- Context Relevancy: How relevant retrieved context is to query
- Answer Relevancy: How relevant answer is to query
- Answer Correctness: Answer accuracy vs ground truth
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_relevancy,
        faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class EvaluationMetric(Enum):
    """Available evaluation metrics"""

    FAITHFULNESS = "faithfulness"
    CONTEXT_RELEVANCY = "context_relevancy"
    ANSWER_RELEVANCY = "answer_relevancy"
    ANSWER_CORRECTNESS = "answer_correctness"
    RETRIEVAL_SIMILARITY = "retrieval_similarity"


@dataclass
class EvaluationResult:
    """Result of RAG evaluation"""

    query: str
    generated_answer: str
    retrieved_contexts: List[str]
    ground_truth: Optional[str]
    metrics: Dict[str, float]
    processing_time: float
    metadata: Dict[str, Any]


class RAGEvaluator:
    """
    Comprehensive RAG evaluation using RAGAs framework

    Based on paper Section A.7 evaluation methodology:
    - Faithfulness: Answer consistency with context
    - Context Relevancy: Context relevance to query
    - Answer Relevancy: Answer relevance to query
    - Answer Correctness: Answer accuracy vs ground truth
    """

    def __init__(
        self,
        evaluation_model: str = "gpt-3.5-turbo",
        embedding_model: str = "BAAI/bge-m3",
    ):
        """
        Initialize the RAG evaluator

        Args:
            evaluation_model: Model for evaluation (GPT-3.5-turbo recommended)
            embedding_model: Model for embeddings
        """
        self.evaluation_model = evaluation_model
        self.embedding_model_name = embedding_model

        # Initialize components
        self.embedding_model = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize evaluation components"""
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")

    def evaluate_sample(
        self,
        query: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single RAG sample

        Args:
            query: Original query
            generated_answer: Generated answer
            retrieved_contexts: Retrieved context documents
            ground_truth: Ground truth answer (optional)

        Returns:
            EvaluationResult with all metrics
        """
        import time

        start_time = time.time()

        metrics = {}

        # Calculate each metric
        if RAGAS_AVAILABLE:
            try:
                # Prepare data for RAGAs
                data = {
                    "question": [query],
                    "answer": [generated_answer],
                    "contexts": [retrieved_contexts],
                }

                if ground_truth:
                    data["ground_truth"] = [ground_truth]

                dataset = Dataset.from_dict(data)

                # Evaluate with RAGAs
                ragas_metrics = [faithfulness, answer_relevancy, context_relevancy]

                if ground_truth:
                    ragas_metrics.append(answer_correctness)

                results = evaluate(dataset, ragas_metrics)

                # Extract scores
                for metric_name in results.keys():
                    if hasattr(results[metric_name], "values"):
                        metrics[metric_name] = float(results[metric_name].values[0])
                    else:
                        metrics[metric_name] = float(results[metric_name])

            except Exception as e:
                print(f"RAGAs evaluation failed: {e}")
                # Fallback to manual calculation
                metrics = self._calculate_manual_metrics(
                    query, generated_answer, retrieved_contexts, ground_truth
                )
        else:
            # Manual calculation if RAGAs not available
            metrics = self._calculate_manual_metrics(
                query, generated_answer, retrieved_contexts, ground_truth
            )

        processing_time = time.time() - start_time

        metadata = {
            "evaluation_model": self.evaluation_model,
            "embedding_model": self.embedding_model_name,
            "ragas_available": RAGAS_AVAILABLE,
            "ground_truth_provided": ground_truth is not None,
            "num_contexts": len(retrieved_contexts),
        }

        return EvaluationResult(
            query=query,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_contexts,
            ground_truth=ground_truth,
            metrics=metrics,
            processing_time=processing_time,
            metadata=metadata,
        )

    def _calculate_manual_metrics(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Manual calculation of metrics when RAGAs is not available

        Args:
            query: Query
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Ground truth answer

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Simple faithfulness calculation
        faithfulness_score = self._calculate_faithfulness(answer, contexts)
        metrics["faithfulness"] = faithfulness_score

        # Simple context relevancy
        context_relevancy_score = self._calculate_context_relevancy(query, contexts)
        metrics["context_relevancy"] = context_relevancy_score

        # Simple answer relevancy
        answer_relevancy_score = self._calculate_answer_relevancy(query, answer)
        metrics["answer_relevancy"] = answer_relevancy_score

        # Answer correctness if ground truth available
        if ground_truth:
            correctness_score = self._calculate_answer_correctness(answer, ground_truth)
            metrics["answer_correctness"] = correctness_score

        return metrics

    def _calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Calculate faithfulness score manually"""
        if not contexts:
            return 0.0

        # Simple keyword overlap (in production, use NLI model)
        answer_words = set(answer.lower().split())
        context_words = set()

        for context in contexts:
            context_words.update(context.lower().split())

        overlap = len(answer_words.intersection(context_words))
        total_answer_words = len(answer_words)

        return overlap / total_answer_words if total_answer_words > 0 else 0.0

    def _calculate_context_relevancy(self, query: str, contexts: List[str]) -> float:
        """Calculate context relevancy score manually"""
        if not contexts:
            return 0.0

        query_words = set(query.lower().split())
        total_relevant_sentences = 0
        total_sentences = 0

        for context in contexts:
            sentences = context.split(".")
            total_sentences += len(sentences)

            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if query_words.intersection(sentence_words):
                    total_relevant_sentences += 1

        return (
            total_relevant_sentences / total_sentences if total_sentences > 0 else 0.0
        )

    def _calculate_answer_relevancy(self, query: str, answer: str) -> float:
        """Calculate answer relevancy score manually"""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        overlap = len(query_words.intersection(answer_words))
        return overlap / len(query_words) if query_words else 0.0

    def _calculate_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """Calculate answer correctness manually"""
        # Simple string similarity (in production, use semantic similarity)
        answer_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())

        intersection = answer_words.intersection(gt_words)
        union = answer_words.union(gt_words)

        return len(intersection) / len(union) if union else 0.0

    def calculate_retrieval_similarity(
        self, retrieved_docs: List[str], gold_docs: List[str]
    ) -> float:
        """
        Calculate cosine similarity between retrieved and gold documents

        Args:
            retrieved_docs: Retrieved documents
            gold_docs: Gold standard documents

        Returns:
            Average cosine similarity
        """
        if not self.embedding_model or not retrieved_docs or not gold_docs:
            return 0.0

        try:
            # Embed all documents
            all_docs = retrieved_docs + gold_docs
            embeddings = self.embedding_model.encode(all_docs)

            retrieved_embeddings = embeddings[: len(retrieved_docs)]
            gold_embeddings = embeddings[len(retrieved_docs) :]

            # Calculate similarities
            similarities = []
            for ret_emb in retrieved_embeddings:
                sims = []
                for gold_emb in gold_embeddings:
                    sim = self.embedding_model.similarity_fn([ret_emb], [gold_emb])[0][
                        0
                    ]
                    sims.append(sim)
                similarities.append(max(sims) if sims else 0.0)

            return sum(similarities) / len(similarities) if similarities else 0.0

        except Exception as e:
            print(f"Retrieval similarity calculation failed: {e}")
            return 0.0

    def evaluate_rag_system(
        self,
        queries: List[str],
        generated_answers: List[str],
        retrieved_contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a complete RAG system

        Args:
            queries: List of queries
            generated_answers: List of generated answers
            retrieved_contexts: List of retrieved context lists
            ground_truths: List of ground truth answers

        Returns:
            Comprehensive evaluation results
        """
        results = []
        for i, query in enumerate(queries):
            answer = generated_answers[i] if i < len(generated_answers) else ""
            contexts = retrieved_contexts[i] if i < len(retrieved_contexts) else []
            ground_truth = (
                ground_truths[i] if ground_truths and i < len(ground_truths) else None
            )

            result = self.evaluate_sample(query, answer, contexts, ground_truth)
            results.append(result)

        # Aggregate results
        aggregated_metrics = {}
        for metric in [
            "faithfulness",
            "context_relevancy",
            "answer_relevancy",
            "answer_correctness",
        ]:
            scores = [
                r.metrics.get(metric, 0.0) for r in results if metric in r.metrics
            ]
            if scores:
                aggregated_metrics[f"{metric}_mean"] = sum(scores) / len(scores)
                aggregated_metrics[f"{metric}_std"] = (
                    sum((s - aggregated_metrics[f"{metric}_mean"]) ** 2 for s in scores)
                    / len(scores)
                ) ** 0.5

        return {
            "individual_results": results,
            "aggregated_metrics": aggregated_metrics,
            "total_samples": len(results),
            "evaluation_time": sum(r.processing_time for r in results),
        }

    def get_expected_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get expected performance ranges from paper evaluation
        Based on paper Section A.7 comprehensive evaluation
        """
        return {
            "faithfulness": {
                "expected_range": "0.7-0.9",
                "description": "Answer consistency with retrieved context",
                "target": 0.85,
            },
            "context_relevancy": {
                "expected_range": "0.6-0.8",
                "description": "Proportion of relevant context sentences",
                "target": 0.75,
            },
            "answer_relevancy": {
                "expected_range": "0.8-0.95",
                "description": "Answer relevance to original query",
                "target": 0.90,
            },
            "answer_correctness": {
                "expected_range": "0.75-0.95",
                "description": "Answer accuracy vs ground truth",
                "target": 0.85,
            },
        }
