#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Chunking Techniques for RAG
Based on EMNLP 2024 Paper Section A.2

Implements multiple chunking strategies:
- Fixed-size chunking (128, 256, 512, 1024, 2048 tokens)
- Small-to-big chunking (175 small + 512 large tokens)
- Sliding window with overlap (20 tokens)
- Semantic chunking with embeddings
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class ChunkingMethod(Enum):
    """Available chunking methods"""

    FIXED = "fixed"
    SMALL_TO_BIG = "small_to_big"
    SLIDING_WINDOW = "sliding_window"
    SEMANTIC = "semantic"


@dataclass
class ChunkingResult:
    """Result of chunking operation"""

    chunks: List[str]
    method: ChunkingMethod
    chunk_sizes: List[int]
    overlaps: List[int]
    metadata: Dict[str, any]
    processing_time: float


class AdvancedChunker:
    """
    Advanced document chunking for RAG systems

    Based on paper configurations:
    - Small chunk size: 175 tokens
    - Large chunk size: 512 tokens
    - Overlap: 20 tokens
    - Embedding model: text-embedding-ada-002 or BAAI/bge-m3
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        tokenizer: str = "cl100k_base",  # GPT-3.5/4 tokenizer
        small_chunk_size: int = 175,
        large_chunk_size: int = 512,
        overlap_tokens: int = 20,
        semantic_threshold: float = 0.7,
    ):
        """
        Initialize the advanced chunker

        Args:
            embedding_model: Sentence transformer model for semantic chunking
            tokenizer: Tokenizer to use for token counting
            small_chunk_size: Size of small chunks (tokens)
            large_chunk_size: Size of large chunks (tokens)
            overlap_tokens: Token overlap between chunks
            semantic_threshold: Similarity threshold for semantic chunking
        """
        self.embedding_model_name = embedding_model
        self.tokenizer_name = tokenizer
        self.small_chunk_size = small_chunk_size
        self.large_chunk_size = large_chunk_size
        self.overlap_tokens = overlap_tokens
        self.semantic_threshold = semantic_threshold

        # Initialize components
        self.tokenizer = None
        self.embedding_model = None

        self._initialize_tokenizer()
        self._initialize_embedding_model()

    def _initialize_tokenizer(self):
        """Initialize the tokenizer for token counting"""
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(self.tokenizer_name)
            except Exception:
                # Fallback to cl100k_base
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            # Fallback tokenization using split
            self.tokenizer = None

    def _initialize_embedding_model(self):
        """Initialize the embedding model for semantic chunking"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                print(
                    f"Warning: Could not load embedding model {self.embedding_model_name}: {e}"
                )
                self.embedding_model = None
        else:
            self.embedding_model = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate 4 characters per token
            return len(text) // 4

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def fixed_size_chunking(
        self, text: str, chunk_size: int, overlap: int = 0
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Fixed-size chunking with optional overlap

        Args:
            text: Input text
            chunk_size: Size of each chunk in tokens
            overlap: Token overlap between chunks

        Returns:
            Tuple of (chunks, chunk_sizes, overlaps)
        """
        if not text.strip():
            return [], [], []

        chunks = []
        chunk_sizes = []
        overlaps = []

        words = text.split()
        current_chunk = []
        current_tokens = 0

        i = 0
        while i < len(words):
            word = words[i]
            word_tokens = self.count_tokens(word)

            # Check if adding this word would exceed chunk size
            if current_tokens + word_tokens > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                chunk_sizes.append(current_tokens)
                overlaps.append(overlap)

                # Start new chunk with overlap
                if overlap > 0:
                    overlap_words = []
                    overlap_tokens = 0
                    for j in range(len(current_chunk) - 1, -1, -1):
                        word_overlap = current_chunk[j]
                        word_overlap_tokens = self.count_tokens(word_overlap)
                        if overlap_tokens + word_overlap_tokens <= overlap:
                            overlap_words.insert(0, word_overlap)
                            overlap_tokens += word_overlap_tokens
                        else:
                            break

                    current_chunk = overlap_words
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0

            # Add current word
            current_chunk.append(word)
            current_tokens += word_tokens
            i += 1

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            chunk_sizes.append(current_tokens)
            overlaps.append(0)  # No overlap for last chunk

        return chunks, chunk_sizes, overlaps

    def small_to_big_chunking(
        self, text: str
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Small-to-big chunking strategy from the paper

        Args:
            text: Input text

        Returns:
            Tuple of (chunks, chunk_sizes, overlaps)
        """
        # First, create small chunks
        small_chunks, _, _ = self.fixed_size_chunking(
            text, self.small_chunk_size, self.overlap_tokens
        )

        # Then create large chunks
        large_chunks, large_sizes, large_overlaps = self.fixed_size_chunking(
            text, self.large_chunk_size, self.overlap_tokens
        )

        # Combine small and large chunks
        all_chunks = small_chunks + large_chunks
        all_sizes = [self.small_chunk_size] * len(small_chunks) + large_sizes
        all_overlaps = [self.overlap_tokens] * len(small_chunks) + large_overlaps

        return all_chunks, all_sizes, all_overlaps

    def sliding_window_chunking(
        self, text: str, window_size: int = 512, stride: int = 256
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Sliding window chunking with configurable stride

        Args:
            text: Input text
            window_size: Size of sliding window
            stride: Step size for sliding window

        Returns:
            Tuple of (chunks, chunk_sizes, overlaps)
        """
        if stride >= window_size:
            raise ValueError("Stride must be less than window size")

        overlap = window_size - stride
        return self.fixed_size_chunking(text, window_size, overlap)

    def semantic_chunking(
        self, text: str, max_chunk_size: int = 512
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Semantic chunking based on sentence embeddings

        Args:
            text: Input text
            max_chunk_size: Maximum chunk size in tokens

        Returns:
            Tuple of (chunks, chunk_sizes, overlaps)
        """
        if not self.embedding_model:
            # Fallback to fixed-size chunking
            return self.fixed_size_chunking(text, max_chunk_size, self.overlap_tokens)

        # Split into sentences
        sentences = self.split_into_sentences(text)
        if not sentences:
            return [], [], []

        # Get embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences)

        chunks = []
        current_chunk_sentences = []
        current_chunk_text = ""
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)

            # Check if adding this sentence would exceed max size
            if (
                current_tokens + sentence_tokens > max_chunk_size
                and current_chunk_sentences
            ):
                # Check semantic similarity with previous sentence
                if i > 0:
                    similarity = cosine_similarity(
                        [embeddings[i - 1]], [embeddings[i]]
                    )[0][0]

                    # If similarity is low, start new chunk
                    if similarity < self.semantic_threshold:
                        chunk_text = " ".join(current_chunk_sentences)
                        chunks.append(chunk_text)
                        current_chunk_sentences = [sentence]
                        current_chunk_text = sentence
                        current_tokens = sentence_tokens
                        continue

                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(chunk_text)
                current_chunk_sentences = [sentence]
                current_chunk_text = sentence
                current_tokens = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_text += " " + sentence if current_chunk_text else sentence
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)

        # Calculate sizes and overlaps
        chunk_sizes = [self.count_tokens(chunk) for chunk in chunks]
        overlaps = [0] * len(chunks)  # Semantic chunking doesn't use fixed overlaps

        return chunks, chunk_sizes, overlaps

    def chunk_document(
        self, text: str, method: ChunkingMethod = ChunkingMethod.SMALL_TO_BIG, **kwargs
    ) -> ChunkingResult:
        """
        Chunk a document using specified method

        Args:
            text: Input document text
            method: Chunking method to use
            **kwargs: Additional parameters for specific methods

        Returns:
            ChunkingResult with chunks and metadata
        """
        import time

        start_time = time.time()

        if method == ChunkingMethod.FIXED:
            chunk_size = kwargs.get("chunk_size", self.large_chunk_size)
            overlap = kwargs.get("overlap", self.overlap_tokens)
            chunks, chunk_sizes, overlaps = self.fixed_size_chunking(
                text, chunk_size, overlap
            )

        elif method == ChunkingMethod.SMALL_TO_BIG:
            chunks, chunk_sizes, overlaps = self.small_to_big_chunking(text)

        elif method == ChunkingMethod.SLIDING_WINDOW:
            window_size = kwargs.get("window_size", self.large_chunk_size)
            stride = kwargs.get("stride", window_size // 2)
            chunks, chunk_sizes, overlaps = self.sliding_window_chunking(
                text, window_size, stride
            )

        elif method == ChunkingMethod.SEMANTIC:
            max_chunk_size = kwargs.get("max_chunk_size", self.large_chunk_size)
            chunks, chunk_sizes, overlaps = self.semantic_chunking(text, max_chunk_size)

        else:
            raise ValueError(f"Unknown chunking method: {method}")

        processing_time = time.time() - start_time

        metadata = {
            "method": method.value,
            "total_chunks": len(chunks),
            "avg_chunk_size": np.mean(chunk_sizes) if chunk_sizes else 0,
            "total_tokens": sum(chunk_sizes),
            "original_text_tokens": self.count_tokens(text),
            **kwargs,
        }

        return ChunkingResult(
            chunks=chunks,
            method=method,
            chunk_sizes=chunk_sizes,
            overlaps=overlaps,
            metadata=metadata,
            processing_time=processing_time,
        )

    def compare_chunking_methods(
        self, text: str, methods: List[ChunkingMethod] = None
    ) -> Dict[str, ChunkingResult]:
        """
        Compare different chunking methods on the same text

        Args:
            text: Input text to chunk
            methods: List of methods to compare (default: all methods)

        Returns:
            Dictionary mapping method names to ChunkingResult
        """
        if methods is None:
            methods = list(ChunkingMethod)

        results = {}
        for method in methods:
            try:
                result = self.chunk_document(text, method)
                results[method.value] = result
            except Exception as e:
                print(f"Error with method {method.value}: {e}")
                continue

        return results

    def get_optimal_chunk_sizes(self, text: str) -> Dict[str, int]:
        """
        Get optimal chunk sizes based on paper experiments

        From paper Table 3: Chunk Size Comparison on lyft_2021 dataset
        - 2048 tokens: 80.37 faithfulness, 91.11 relevancy
        - 1024 tokens: 94.26 faithfulness, 95.56 relevancy
        - 512 tokens: 97.59 faithfulness, 97.41 relevancy
        - 256 tokens: 97.22 faithfulness, 97.78 relevancy
        - 128 tokens: 95.74 faithfulness, 97.22 relevancy

        Returns:
            Dictionary with optimal sizes for different metrics
        """
        return {
            "best_faithfulness": 512,  # 97.59 faithfulness
            "best_relevancy": 256,  # 97.78 relevancy
            "best_overall": 512,  # Good balance
            "small_chunk": 175,  # For small-to-big
            "large_chunk": 512,  # For small-to-big
            "overlap": 20,  # Standard overlap
        }
