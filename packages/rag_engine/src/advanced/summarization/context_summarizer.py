#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Summarization Methods for RAG
Based on EMNLP 2024 Paper Section A.5

Implements multiple summarization strategies:
- Selective Context: Self-information based compression
- LongLLMLingua: Instruction-following compression
- Recomp: Extractive + Abstractive summarization
"""

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class SummarizationMethod(Enum):
    """Available summarization methods"""

    SELECTIVE_CONTEXT = "selective_context"
    LONGLM_LINGUA = "longllm_lingua"
    RECOMP = "recomp"


@dataclass
class SummarizationResult:
    """Result of context summarization"""

    original_text: str
    summarized_text: str
    compression_ratio: float
    method: SummarizationMethod
    processing_time: float
    metadata: Dict[str, Any]


class ContextSummarizer:
    """
    Multi-method context summarizer for RAG systems

    Based on paper Table 11 results:
    - Selective Context: F1=25.05 (65 tokens), best for abstractive
    - LongLLMLingua: F1=21.32 (51 tokens), good compression
    - Recomp (abstractive): F1=33.68 (59 tokens), best overall
    """

    def __init__(
        self,
        compression_model: str = "microsoft/DialoGPT-small",
        summarization_ratio: float = 0.4,
        device: str = "auto",
    ):
        """
        Initialize the context summarizer

        Args:
            compression_model: Model for compression tasks
            summarization_ratio: Target compression ratio
            device: Device to run on
        """
        self.compression_model_name = compression_model
        self.summarization_ratio = summarization_ratio
        self.device = (
            device if device != "auto" else ("cuda" if self._has_cuda() else "cpu")
        )

        # Initialize components
        self.tokenizer = None
        self.compression_model = None
        self.summarization_pipeline = None

        self._initialize_components()

    def _has_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _initialize_components(self):
        """Initialize required components"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Initialize tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "gpt2"
                )  # Simple tokenizer
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Initialize compression pipeline
                self.summarization_pipeline = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if self.device == "cuda" else -1,
                )

            except Exception as e:
                print(f"Warning: Could not initialize summarization components: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())  # Fallback

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass

        # Fallback sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def selective_context_compression(
        self, text: str, query: Optional[str] = None
    ) -> str:
        """
        Selective Context compression using self-information

        Args:
            text: Text to compress
            query: Optional query for query-aware compression

        Returns:
            Compressed text
        """
        sentences = self.split_into_sentences(text)
        if not sentences:
            return text

        # Calculate self-information scores for each sentence
        sentence_scores = []

        for sentence in sentences:
            # Self-information based on word frequency
            words = re.findall(r"\b\w+\b", sentence.lower())
            word_freq = Counter(words)

            # Calculate information score (rarer words = higher information)
            info_score = 0.0
            for word in words:
                # Simple frequency-based scoring
                freq = word_freq[word]
                info_score += 1.0 / (freq + 1)  # Higher score for rarer words

            # Normalize by sentence length
            info_score /= len(words) if words else 1

            sentence_scores.append((sentence, info_score))

        # Sort by information score (descending)
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top sentences based on compression ratio
        target_sentences = max(1, int(len(sentences) * self.summarization_ratio))
        selected_sentences = sentence_scores[:target_sentences]

        # Sort back to original order for coherence
        original_order = []
        selected_texts = [s[0] for s in selected_sentences]

        for sentence in sentences:
            if sentence in selected_texts:
                original_order.append(sentence)

        return " ".join(original_order)

    def longllm_lingua_compression(self, text: str, query: Optional[str] = None) -> str:
        """
        LongLLMLingua compression (instruction-following)

        Args:
            text: Text to compress
            query: Optional query context

        Returns:
            Compressed text
        """
        # Simplified LongLLMLingua implementation
        # In production, this would use the actual LongLLMLingua model

        sentences = self.split_into_sentences(text)
        if not sentences:
            return text

        # Score sentences based on length and position
        scored_sentences = []

        for i, sentence in enumerate(sentences):
            # Prefer sentences that are:
            # 1. Not too short or long
            # 2. Early in the document (often contain key information)
            # 3. Contain question words if query is provided

            length_score = (
                1.0 - abs(len(sentence.split()) - 15) / 30
            )  # Optimal ~15 words
            position_score = 1.0 / (i + 1)  # Earlier sentences preferred

            query_score = 0.0
            if query:
                query_words = set(query.lower().split())
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                query_score = overlap / len(query_words) if query_words else 0

            total_score = 0.4 * length_score + 0.4 * position_score + 0.2 * query_score
            scored_sentences.append((sentence, total_score))

        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        target_count = max(1, int(len(sentences) * self.summarization_ratio))
        selected_sentences = [s[0] for s in scored_sentences[:target_count]]

        # Return in original order
        result_sentences = []
        for sentence in sentences:
            if sentence in selected_sentences:
                result_sentences.append(sentence)

        return " ".join(result_sentences)

    def recomp_summarization(self, text: str, query: Optional[str] = None) -> str:
        """
        Recomp: Extractive + Abstractive summarization

        Args:
            text: Text to summarize
            query: Optional query context

        Returns:
            Summarized text
        """
        # Phase 1: Extractive summarization (select key sentences)
        sentences = self.split_into_sentences(text)
        if len(sentences) <= 3:
            extractive_summary = text
        else:
            # Select top sentences using TF-IDF like scoring
            extractive_summary = self._extractive_summarization(sentences, query)

        # Phase 2: Abstractive summarization (if pipeline available)
        if self.summarization_pipeline and len(extractive_summary.split()) > 50:
            try:
                # Use the summarization pipeline
                summary_result = self.summarization_pipeline(
                    extractive_summary,
                    max_length=int(len(extractive_summary.split()) * 0.6),
                    min_length=int(len(extractive_summary.split()) * 0.3),
                    do_sample=False,
                )
                return summary_result[0]["summary_text"]
            except Exception as e:
                print(f"Abstractive summarization failed: {e}")

        return extractive_summary

    def _extractive_summarization(
        self, sentences: List[str], query: Optional[str] = None
    ) -> str:
        """Extractive summarization using sentence scoring"""
        if not sentences:
            return ""

        # Calculate TF-IDF like scores
        all_words = []
        for sentence in sentences:
            words = re.findall(r"\b\w+\b", sentence.lower())
            all_words.extend(words)

        word_freq = Counter(all_words)
        total_sentences = len(sentences)

        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = re.findall(r"\b\w+\b", sentence.lower())
            score = 0.0

            for word in words:
                # TF-IDF score
                tf = words.count(word) / len(words) if words else 0
                idf = np.log(total_sentences / (word_freq[word] + 1))
                score += tf * idf

            # Position bonus (earlier sentences often more important)
            position_bonus = 1.0 / (i + 1)
            score += position_bonus

            # Query relevance bonus
            if query:
                query_words = set(query.lower().split())
                sentence_words = set(words)
                overlap = len(query_words.intersection(sentence_words))
                query_bonus = overlap * 2.0
                score += query_bonus

            sentence_scores.append((sentence, score))

        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        target_count = max(1, int(len(sentences) * self.summarization_ratio))
        selected_sentences = [s[0] for s in sentence_scores[:target_count]]

        # Return in original order for coherence
        result = []
        for sentence in sentences:
            if sentence in selected_sentences:
                result.append(sentence)

        return " ".join(result)

    def summarize(
        self,
        text: str,
        method: SummarizationMethod = SummarizationMethod.RECOMP,
        query: Optional[str] = None,
    ) -> SummarizationResult:
        """
        Summarize text using specified method

        Args:
            text: Text to summarize
            method: Summarization method to use
            query: Optional query for context-aware summarization

        Returns:
            SummarizationResult
        """
        import time

        start_time = time.time()

        original_tokens = self.count_tokens(text)

        if method == SummarizationMethod.SELECTIVE_CONTEXT:
            summarized_text = self.selective_context_compression(text, query)
        elif method == SummarizationMethod.LONGLM_LINGUA:
            summarized_text = self.longllm_lingua_compression(text, query)
        elif method == SummarizationMethod.RECOMP:
            summarized_text = self.recomp_summarization(text, query)
        else:
            summarized_text = text  # No summarization

        summarized_tokens = self.count_tokens(summarized_text)
        compression_ratio = (
            summarized_tokens / original_tokens if original_tokens > 0 else 1.0
        )

        processing_time = time.time() - start_time

        metadata = {
            "method": method.value,
            "original_tokens": original_tokens,
            "summarized_tokens": summarized_tokens,
            "compression_ratio": compression_ratio,
            "query_provided": query is not None,
            "target_ratio": self.summarization_ratio,
        }

        return SummarizationResult(
            original_text=text,
            summarized_text=summarized_text,
            compression_ratio=compression_ratio,
            method=method,
            processing_time=processing_time,
            metadata=metadata,
        )

    def batch_summarize(
        self,
        texts: List[str],
        method: SummarizationMethod = SummarizationMethod.RECOMP,
        queries: Optional[List[str]] = None,
    ) -> List[SummarizationResult]:
        """
        Summarize multiple texts

        Args:
            texts: List of texts to summarize
            method: Summarization method
            queries: Optional list of queries (one per text)

        Returns:
            List of SummarizationResult objects
        """
        results = []
        for i, text in enumerate(texts):
            query = queries[i] if queries and i < len(queries) else None
            result = self.summarize(text, method, query)
            results.append(result)

        return results

    def get_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for summarization methods
        Based on paper Table 11 results
        """
        return {
            SummarizationMethod.SELECTIVE_CONTEXT.value: {
                "f1_nq": 25.05,
                "f1_triviaqa": 34.25,
                "f1_hotpotqa": 34.43,
                "avg_f1": 31.24,
                "avg_tokens": 67,
                "description": "Self-information based, best abstractive performance",
            },
            SummarizationMethod.LONGLM_LINGUA.value: {
                "f1_nq": 21.32,
                "f1_triviaqa": 32.81,
                "f1_hotpotqa": 30.79,
                "avg_f1": 28.29,
                "avg_tokens": 55,
                "description": "Good compression, moderate performance",
            },
            SummarizationMethod.RECOMP.value: {
                "f1_nq": 33.68,
                "f1_triviaqa": 35.87,
                "f1_hotpotqa": 29.01,
                "avg_f1": 32.85,
                "avg_tokens": 59,
                "description": "Best overall performance, extractive + abstractive",
            },
        }
