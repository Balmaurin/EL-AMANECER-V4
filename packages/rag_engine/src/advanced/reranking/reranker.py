#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reranking Systems for RAG
Based on EMNLP 2024 Paper Section A.4

Implements multiple reranking models:
- RankLLaMA (7B) - Best performance: MRR@1=22.08, latency=82.4s
- MonoT5 - Balanced: MRR@1=21.62, latency=4.5s
- TILDEv2 - Fastest: MRR@1=18.57, latency=0.02s
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import ndcg_score

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        LlamaForSequenceClassification,
        LlamaTokenizer,
        T5ForConditionalGeneration,
        T5Tokenizer,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RerankerType(Enum):
    """Available reranker types"""

    RANKLLAMA = "rankllama"
    MONOT5 = "monot5"
    TILDEV2 = "tildev2"


@dataclass
class RerankingResult:
    """Result of reranking operation"""

    query: str
    original_docs: List[str]
    reranked_docs: List[str]
    original_scores: List[float]
    reranked_scores: List[float]
    reranker_type: RerankerType
    processing_time: float
    metadata: Dict[str, Any]


class Reranker:
    """
    Multi-model reranker supporting RankLLaMA, MonoT5, and TILDEv2

    Based on paper results (Table 10):
    - RankLLaMA: Best performance (MRR@1=22.08) but slowest (82.4s)
    - MonoT5: Balanced performance (MRR@1=21.62) and latency (4.5s)
    - TILDEv2: Fastest (0.02s) but lower performance (MRR@1=18.57)
    """

    def __init__(
        self,
        reranker_type: RerankerType = RerankerType.MONOT5,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize the reranker

        Args:
            reranker_type: Type of reranker to use
            model_path: Custom model path (optional)
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.reranker_type = reranker_type
        self.model_path = model_path
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate model based on reranker type"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, reranker will not function")
            return

        try:
            if self.reranker_type == RerankerType.RANKLLAMA:
                self._initialize_rankllama()
            elif self.reranker_type == RerankerType.MONOT5:
                self._initialize_monot5()
            elif self.reranker_type == RerankerType.TILDEV2:
                self._initialize_tildev2()
        except Exception as e:
            logger.error(f"Failed to initialize {self.reranker_type.value} model: {e}")

    def _initialize_rankllama(self):
        """Initialize RankLLaMA model (Llama-2-7b based)"""
        model_name = self.model_path or "meta-llama/Llama-2-7b-hf"

        logger.info(f"Initializing RankLLaMA from {model_name}")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,  # Regression for ranking scores
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)

    def _initialize_monot5(self):
        """Initialize MonoT5 model (T5-base based)"""
        model_name = self.model_path or "castorini/monot5-base-msmarco"

        logger.info(f"Initializing MonoT5 from {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.model.to(self.device)

    def _initialize_tildev2(self):
        """Initialize TILDEv2 model (BERT-base based)"""
        model_name = self.model_path or "ielab/TILDEv2"

        logger.info(f"Initializing TILDEv2 from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )

        self.model.to(self.device)

    def _rankllama_score(self, query: str, doc: str) -> float:
        """
        Score query-doc pair using RankLLaMA

        Args:
            query: Query text
            doc: Document text

        Returns:
            Relevance score
        """
        if not self.model or not self.tokenizer:
            return 0.0

        # Prepare input (following RankLLaMA format)
        input_text = f"Query: {query} Document: {doc}"

        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits.squeeze().item()

        return score

    def _monot5_score(self, query: str, doc: str) -> float:
        """
        Score query-doc pair using MonoT5

        Args:
            query: Query text
            doc: Document text

        Returns:
            Relevance score (1 for relevant, 0 for not)
        """
        if not self.model or not self.tokenizer:
            return 0.0

        # Prepare input in T5 format
        input_text = f"Query: {query} Document: {doc} Relevant:"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # Get the score for "true" token
            logits = outputs.scores[0][0]  # First generated token logits
            true_token_id = self.tokenizer.encode("true")[0]
            false_token_id = self.tokenizer.encode("false")[0]

            true_score = logits[true_token_id].item()
            false_score = logits[false_token_id].item()

            # Convert to probability
            score = 1 / (1 + np.exp(false_score - true_score))

        return score

    def _tildev2_score(self, query: str, doc: str) -> float:
        """
        Score query-doc pair using TILDEv2

        Args:
            query: Query text
            doc: Document text

        Returns:
            Relevance score
        """
        if not self.model or not self.tokenizer:
            return 0.0

        # TILDEv2 uses a special format
        input_text = f"{query} {self.tokenizer.sep_token} {doc}"

        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits.squeeze().item()

        return score

    def score_document(self, query: str, doc: str) -> float:
        """
        Score a single query-document pair

        Args:
            query: Query text
            doc: Document text

        Returns:
            Relevance score
        """
        if self.reranker_type == RerankerType.RANKLLAMA:
            return self._rankllama_score(query, doc)
        elif self.reranker_type == RerankerType.MONOT5:
            return self._monot5_score(query, doc)
        elif self.reranker_type == RerankerType.TILDEV2:
            return self._tildev2_score(query, doc)
        else:
            return 0.0

    def rerank(
        self,
        query: str,
        documents: List[str],
        original_scores: Optional[List[float]] = None,
    ) -> RerankingResult:
        """
        Rerank a list of documents for a query

        Args:
            query: Query text
            documents: List of document texts
            original_scores: Original retrieval scores (optional)

        Returns:
            RerankingResult with reranked documents
        """
        import time

        start_time = time.time()

        if not documents:
            return RerankingResult(
                query=query,
                original_docs=[],
                reranked_docs=[],
                original_scores=[],
                reranked_scores=[],
                reranker_type=self.reranker_type,
                processing_time=time.time() - start_time,
                metadata={},
            )

        # Score all documents
        reranked_scores = []
        for doc in documents:
            score = self.score_document(query, doc)
            reranked_scores.append(score)

        # Sort by reranking scores (descending)
        scored_docs = list(zip(documents, reranked_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_docs = [doc for doc, _ in scored_docs]
        reranked_scores_sorted = [score for _, score in scored_docs]

        # Use original scores if provided, otherwise use document indices
        if original_scores is None:
            original_scores = list(range(len(documents)))

        processing_time = time.time() - start_time

        metadata = {
            "reranker_type": self.reranker_type.value,
            "num_documents": len(documents),
            "device": self.device,
            "model_loaded": self.model is not None,
            "performance_metrics": self.get_performance_metrics(),
        }

        return RerankingResult(
            query=query,
            original_docs=documents,
            reranked_docs=reranked_docs,
            original_scores=original_scores,
            reranked_scores=reranked_scores_sorted,
            reranker_type=self.reranker_type,
            processing_time=processing_time,
            metadata=metadata,
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the current reranker type
        Based on paper Table 10 results
        """
        metrics = {
            RerankerType.RANKLLAMA: {
                "mrr@1": 22.08,
                "mrr@10": 32.35,
                "mrr@1k": 32.97,
                "hit_rate@10": 54.53,
                "latency_seconds": 82.4,
                "model_size": "7B parameters",
                "description": "Best performance, highest latency",
            },
            RerankerType.MONOT5: {
                "mrr@1": 21.62,
                "mrr@10": 31.78,
                "mrr@1k": 32.40,
                "hit_rate@10": 54.07,
                "latency_seconds": 4.5,
                "model_size": "220M parameters",
                "description": "Balanced performance and latency",
            },
            RerankerType.TILDEV2: {
                "mrr@1": 18.57,
                "mrr@10": 27.83,
                "mrr@1k": 28.60,
                "hit_rate@10": 49.07,
                "latency_seconds": 0.02,
                "model_size": "110M parameters",
                "description": "Fastest, requires pre-indexed collection",
            },
        }

        return metrics.get(self.reranker_type, {})

    def calculate_ndcg_improvement(
        self,
        original_ranking: List[int],
        reranked_order: List[int],
        relevance_labels: List[int],
    ) -> float:
        """
        Calculate NDCG improvement after reranking

        Args:
            original_ranking: Original document indices
            reranked_order: Reranked document indices
            relevance_labels: Ground truth relevance labels

        Returns:
            NDCG improvement percentage
        """
        if len(relevance_labels) != len(original_ranking):
            return 0.0

        # Calculate NDCG for original ranking
        original_ndcg = ndcg_score(
            [relevance_labels], [relevance_labels[i] for i in original_ranking], k=10
        )

        # Calculate NDCG for reranked order
        reranked_ndcg = ndcg_score(
            [relevance_labels], [relevance_labels[i] for i in reranked_order], k=10
        )

        if original_ndcg == 0:
            return 0.0

        improvement = ((reranked_ndcg - original_ndcg) / original_ndcg) * 100
        return improvement

    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[str]],
        original_scores_list: Optional[List[List[float]]] = None,
    ) -> List[RerankingResult]:
        """
        Rerank multiple query-document sets

        Args:
            queries: List of queries
            document_lists: List of document lists (one per query)
            original_scores_list: List of original score lists

        Returns:
            List of RerankingResult objects
        """
        results = []
        for i, query in enumerate(queries):
            docs = document_lists[i] if i < len(document_lists) else []
            scores = (
                original_scores_list[i]
                if original_scores_list and i < len(original_scores_list)
                else None
            )

            result = self.rerank(query, docs, scores)
            results.append(result)

        return results
