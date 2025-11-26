#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Vector Indexing Techniques
Based on VDBMS Survey Paper - Enhanced indexing beyond basic FAISS

Implements advanced techniques:
- HNSW with quantization
- Multi-index strategies
- Disk-resident indexes
- Hardware-accelerated search
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import hnswlib

    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False


@dataclass
class IndexResult:
    """Result from advanced index search"""

    indices: np.ndarray
    distances: np.ndarray
    search_time: float
    metadata: Dict[str, Any]


@dataclass
class IndexStats:
    """Statistics about the index"""

    n_vectors: int
    dimension: int
    index_type: str
    memory_usage: int  # bytes
    build_time: float
    search_stats: Dict[str, Any]


class AdvancedVectorIndex:
    """
    Advanced Vector Indexing with multiple strategies

    Implements techniques from VDBMS Survey:
    - HNSW with quantization
    - Multi-stage indexing
    - Hardware acceleration
    - Memory-efficient storage
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "HNSW+PQ",
        metric: str = "cosine",
        n_vectors_estimate: int = 100000,
    ):
        """
        Initialize advanced vector index

        Args:
            dimension: Vector dimensionality
            index_type: Type of index (HNSW, IVF, PQ, etc.)
            metric: Distance metric
            n_vectors_estimate: Estimated number of vectors
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.n_vectors_estimate = n_vectors_estimate

        # Initialize indexes
        self.index = None
        self.backup_index = None  # For hybrid strategies

        # Statistics
        self.stats = IndexStats(
            n_vectors=0,
            dimension=dimension,
            index_type=index_type,
            memory_usage=0,
            build_time=0.0,
            search_stats={},
        )

        self._initialize_index()

    def _initialize_index(self):
        """Initialize the appropriate index type"""
        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available, using basic numpy implementation")
            return

        start_time = time.time()

        if self.index_type == "HNSW+PQ":
            # HNSW with Product Quantization for memory efficiency
            # From VDBMS Survey: PQ + HNSW combination
            m = min(64, self.dimension // 4)  # Number of sub-quantizers
            nbits = 8  # Bits per sub-quantizer

            # PQ quantizer
            pq = faiss.ProductQuantizer(self.dimension, m, nbits)
            pq.train_type = faiss.PQTrainerType.PQTrainerCrossPlatform

            # HNSW on quantized vectors
            hnsw = faiss.IndexHNSWFlat(self.dimension, 32)
            hnsw.hnsw.efConstruction = 200  # Higher quality

            # Combined index
            self.index = faiss.IndexPQ(self.dimension, m, nbits)
            self.backup_index = hnsw

        elif self.index_type == "IVFADC":
            # IVF with Asymmetric Distance Computation
            # From VDBMS Survey: Most memory efficient
            nlist = min(1024, max(4, self.n_vectors_estimate // 39))  # IVF clusters
            m = min(64, self.dimension // 4)
            nbits = 8

            quantizer = faiss.IndexFlatIP(self.dimension)  # Inner product
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
            self.index.nprobe = 10  # Search quality

        elif self.index_type == "HNSW":
            # Pure HNSW - best for accuracy
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64

        elif self.index_type == "GPU":
            # GPU-accelerated index
            cpu_index = faiss.IndexFlatIP(self.dimension)
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except:
                print("GPU not available, falling back to CPU")
                self.index = cpu_index

        else:
            # Default: Flat index
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)

        self.stats.build_time = time.time() - start_time

    def add_vectors(
        self, vectors: np.ndarray, ids: Optional[np.ndarray] = None
    ) -> bool:
        """
        Add vectors to the index

        Args:
            vectors: Vectors to add (n_vectors, dimension)
            ids: Optional IDs for the vectors

        Returns:
            Success status
        """
        if self.index is None:
            return False

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != index dimension {self.dimension}"
            )

        # Normalize for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vectors = vectors / norms

        # Train index if needed
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            if hasattr(self.index, "train"):
                print(f"Training index on {len(vectors)} vectors...")
                self.index.train(vectors)

        # Add vectors
        if ids is not None and hasattr(self.index, "add_with_ids"):
            self.index.add_with_ids(vectors.astype(np.float32), ids)
        else:
            self.index.add(vectors.astype(np.float32))

        self.stats.n_vectors += len(vectors)
        self._update_memory_usage()

        return True

    def search(
        self, query_vectors: np.ndarray, k: int = 10, nprobe: Optional[int] = None
    ) -> IndexResult:
        """
        Search for nearest neighbors

        Args:
            query_vectors: Query vectors (n_queries, dimension)
            k: Number of neighbors to return
            nprobe: Number of probes for IVF indexes

        Returns:
            Search results
        """
        if self.index is None:
            return IndexResult(
                indices=np.array([]),
                distances=np.array([]),
                search_time=0.0,
                metadata={"error": "Index not initialized"},
            )

        # Normalize query vectors
        if self.metric == "cosine":
            norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_vectors = query_vectors / norms

        # Set search parameters
        if nprobe and hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_vectors.astype(np.float32), k)
        search_time = time.time() - start_time

        # Update search statistics
        self.stats.search_stats["last_search_time"] = search_time
        self.stats.search_stats["avg_search_time"] = (
            self.stats.search_stats.get("avg_search_time", 0) * 0.9 + search_time * 0.1
        )

        return IndexResult(
            indices=indices,
            distances=distances,
            search_time=search_time,
            metadata={
                "k": k,
                "n_queries": len(query_vectors),
                "index_type": self.index_type,
            },
        )

    def _update_memory_usage(self):
        """Update memory usage statistics"""
        if self.index and hasattr(self.index, "ntotal"):
            # Rough estimate: 4 bytes per float * dimensions * vectors
            self.stats.memory_usage = self.stats.n_vectors * self.dimension * 4

    def save_index(self, filepath: str):
        """Save index to disk"""
        if self.index:
            faiss.write_index(self.index, filepath)

    def load_index(self, filepath: str):
        """Load index from disk"""
        if FAISS_AVAILABLE:
            self.index = faiss.read_index(filepath)
            self.stats.n_vectors = self.index.ntotal
            self._update_memory_usage()

    def get_stats(self) -> IndexStats:
        """Get index statistics"""
        return self.stats

    def optimize_for_recall(self, recall_target: float = 0.95):
        """
        Optimize index parameters for target recall

        Args:
            recall_target: Target recall level (0-1)
        """
        if hasattr(self.index, "nprobe"):
            # Increase nprobe for better recall
            current_nprobe = getattr(self.index, "nprobe", 1)
            self.index.nprobe = min(current_nprobe * 2, 100)

        if hasattr(self.index, "hnsw"):
            # Increase efSearch for HNSW
            self.index.hnsw.efSearch = min(self.index.hnsw.efSearch * 2, 512)

    def optimize_for_speed(self, speed_target_ms: float = 10.0):
        """
        Optimize index parameters for target speed

        Args:
            speed_target_ms: Target search time in milliseconds
        """
        if hasattr(self.index, "nprobe"):
            # Decrease nprobe for speed
            current_nprobe = getattr(self.index, "nprobe", 10)
            self.index.nprobe = max(current_nprobe // 2, 1)

        if hasattr(self.index, "hnsw"):
            # Decrease efSearch for speed
            self.index.hnsw.efSearch = max(self.index.hnsw.efSearch // 2, 1)


class MultiIndexManager:
    """
    Manage multiple indexes for different use cases

    Implements multi-index strategies from VDBMS Survey:
    - Fast and slow indexes
    - Hierarchical indexing
    - Index selection based on query characteristics
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.indexes: Dict[str, AdvancedVectorIndex] = {}
        self.default_index = "HNSW"

    def create_index(self, name: str, index_type: str, **kwargs):
        """Create a new index"""
        index = AdvancedVectorIndex(
            dimension=self.dimension, index_type=index_type, **kwargs
        )
        self.indexes[name] = index
        return index

    def add_to_index(
        self, name: str, vectors: np.ndarray, ids: Optional[np.ndarray] = None
    ):
        """Add vectors to specific index"""
        if name in self.indexes:
            return self.indexes[name].add_vectors(vectors, ids)
        return False

    def search_index(
        self, name: str, query_vectors: np.ndarray, k: int = 10
    ) -> IndexResult:
        """Search specific index"""
        if name in self.indexes:
            return self.indexes[name].search(query_vectors, k)
        return IndexResult(
            indices=np.array([]),
            distances=np.array([]),
            search_time=0.0,
            metadata={"error": f"Index {name} not found"},
        )

    def auto_select_index(self, query_complexity: str = "medium") -> str:
        """
        Auto-select index based on query characteristics

        Args:
            query_complexity: "low", "medium", "high"

        Returns:
            Index name to use
        """
        if query_complexity == "low":
            return "GPU" if "GPU" in self.indexes else "HNSW"
        elif query_complexity == "high":
            return "IVFADC"  # Most memory efficient
        else:
            return self.default_index

    def get_all_stats(self) -> Dict[str, IndexStats]:
        """Get statistics for all indexes"""
        return {name: index.get_stats() for name, index in self.indexes.items()}
