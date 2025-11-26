#!/usr/bin/env python3
"""
RAG SERVICE - Retrieval-Augmented Generation System
===================================================

Enterprise-grade RAG system integrating:
- Sentence-Transformers embeddings (all-MiniLM-L6-v2)
- Vector search with HNSW (FAISS)
- Hybrid retrieval with fallback
- Full integration with ChatService
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Fallback imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("rag_service")


class RAGService:
    """
    Enterprise RAG System with Transformer Embeddings and FAISS
    """

    def __init__(self, config_path: str = "config/universal.yaml", data_dir: str = "./rag_data"):
        self.config_path = config_path
        self.data_dir = Path(data_dir)
        self.config = None

        # System Components
        self.embedding_model = None
        self.documents = []
        self.doc_ids = []
        self.metadatas = []
        self.embeddings = None

        # Fallback TF-IDF
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        # Vector Indices (HNSW/FAISS)
        self.vector_index = None

        # State
        self.initialized = False
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension

        logger.info("🚀 RAG Service initialized")

    async def initialize(self) -> bool:
        """Initialize the complete RAG system"""

        try:
            # Load configuration
            await self._load_config()

            # Initialize embeddings
            success = await self._initialize_embeddings()
            if not success:
                logger.warning("⚠️ Embeddings not available - using TF-IDF fallback")
                await self._initialize_fallback()

            # Load data if exists
            await self._load_data()

            # Initialize indices if data exists
            if self.documents:
                await self._build_indices()

            self.initialized = True
            logger.info(f"✅ RAG Service operational - {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"❌ Error initializing RAG Service: {e}")
            return False

    async def _load_config(self) -> None:
        """Load configuration"""
        try:
            import yaml
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
                    logger.info("✅ Configuration loaded")
            else:
                self.config = {}
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            self.config = {}

    async def _initialize_embeddings(self) -> bool:
        """Initialize Sentence Transformers embeddings"""

        if not ST_AVAILABLE:
            logger.warning("Sentence-Transformers not available")
            return False

        try:
            # Config from universal.yaml or defaults
            embedder_config = self.config.get("embedder", {}) if self.config else {}
            model_name = embedder_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")

            # Force CPU for stability on Windows unless configured otherwise
            device = embedder_config.get("device", "cpu")
            if device != "cpu":
                device = "cpu"  # Force CPU for safety

            logger.info(f"🔧 Loading embedding model: {model_name} on {device}")

            # Load model with cache optimization
            self.embedding_model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=os.environ.get("HF_HOME", "./cache")
            )

            # Configure dimension
            test_embed = self.embedding_model.encode(["test"])
            self.embedding_dimension = len(test_embed[0])

            logger.info(f"✅ Embedding model loaded - dimension: {self.embedding_dimension}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}")
            return False

    async def _initialize_fallback(self) -> None:
        """Initialize TF-IDF fallback"""

        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words=None,
                max_features=1000,
                ngram_range=(1, 2)
            )
            logger.info("✅ TF-IDF Fallback activated")
        else:
            logger.warning("Neither embeddings nor TF-IDF available")

    async def _load_data(self) -> None:
        """Load persisted data"""

        try:
            data_file = self.data_dir / "documents.json"
            if data_file.exists():
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.doc_ids = data.get("doc_ids", [])
                    self.metadatas = data.get("metadatas", [])
                    logger.info(f"✅ Data loaded: {len(self.documents)} documents")
            else:
                logger.info("No pre-loaded data found")

        except Exception as e:
            logger.error(f"Error loading data: {e}")

    async def _build_indices(self) -> None:
        """Build search indices"""

        try:
            if not self.documents:
                return

            # Generate embeddings if model available
            if self.embedding_model:
                logger.info("🔧 Generating embeddings...")

                # Optimized batch processing
                batch_size = min(32, len(self.documents))
                embeddings_list = []

                for i in range(0, len(self.documents), batch_size):
                    batch_texts = self.documents[i:i+batch_size]
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=8
                    )
                    embeddings_list.append(batch_embeddings)

                self.embeddings = np.vstack(embeddings_list)
                logger.info(f"✅ Embeddings generated: {self.embeddings.shape}")

                # Build HNSW index
                await self._build_vector_index()

            # Fallback TF-IDF
            elif self.tfidf_vectorizer and SKLEARN_AVAILABLE:
                logger.info("🔧 Building TF-IDF index...")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
                logger.info("✅ TF-IDF index built")

        except Exception as e:
            logger.error(f"Error building indices: {e}")

    async def _build_vector_index(self) -> None:
        """Build HNSW vector index"""

        if not FAISS_AVAILABLE or self.embeddings is None:
            logger.info("FAISS not available - using manual vector search")
            return

        try:
            dimension = self.embedding_dimension

            # Create HNSW index
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per element
            index.hnsw.efConstruction = 200
            index.add(self.embeddings.astype('float32'))

            self.vector_index = index
            logger.info(f"✅ HNSW Index built: {len(self.embeddings)} vectors")

        except Exception as e:
            logger.error(f"Error building HNSW: {e}")
            self.vector_index = None

    # ========== CHAT SERVICE COMPATIBILITY ==========

    async def index_documents(self, docs: List[str], ids: List[str] = None, metadatas: List[Dict] = None) -> Dict[str, Any]:
        """Index documents - compatible with ChatService"""

        if not docs:
            return {"error": "no documents"}

        if not ids:
            ids = [f"doc_{i}" for i in range(len(docs))]
        if not metadatas:
            metadatas = [{}] * len(docs)

        self.documents = docs
        self.doc_ids = ids
        self.metadatas = metadatas

        # Build indices
        await self._build_indices()

        # Save data
        await self._save_data()

        return {
            "indexed": len(docs),
            "method": "sentence_transformers_hnsw" if self.embedding_model else "tfidf_fallback",
            "embeddings_available": self.embedding_model is not None,
            "vector_index": self.vector_index is not None
        }

    async def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Vector search"""

        if not self.initialized:
            return {"error": "RAG not initialized"}

        if not self.documents:
            return {"error": "no documents indexed"}

        try:
            results = []

            # Method 1: Vector search with embeddings
            if self.embedding_model and self.embeddings is not None:
                results = await self._vector_search(query, top_k)

            # Method 2: Fallback TF-IDF
            elif self.tfidf_vectorizer and self.tfidf_matrix is not None:
                results = await self._tfidf_search(query, top_k)

            # Method 3: Keyword fallback
            else:
                results = await self._keyword_search(query, top_k)

            return {
                "query": query,
                "results": results,
                "method": "sentence_transformers_hnsw" if self.embedding_model else "fallback_tfidf",
                "total_docs": len(self.documents)
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e)}

    async def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Vector search implementation"""

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        query_embedding = query_embedding.astype('float32')

        # FAISS Search
        if self.vector_index is not None:
            faiss.cvar.hnsw.efSearch.set(100)
            distances, indices = self.vector_index.search(
                query_embedding.reshape(1, -1),
                top_k
            )

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    results.append({
                        "document": self.documents[idx],
                        "id": self.doc_ids[idx],
                        "metadata": self.metadatas[idx],
                        "similarity": float(1 - dist),
                        "score": float(1 - dist)
                    })
        else:
            # Manual Cosine Search
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    "document": self.documents[idx],
                    "id": self.doc_ids[idx],
                    "metadata": self.metadatas[idx],
                    "similarity": float(similarities[idx]),
                    "score": float(similarities[idx])
                })

        return results

    async def _tfidf_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """TF-IDF Fallback search"""
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "id": self.doc_ids[idx],
                "metadata": self.metadatas[idx],
                "similarity": float(similarities[idx]),
                "score": float(similarities[idx])
            })
        return results

    async def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Keyword Fallback search"""
        query_lower = query.lower()
        query_keywords = set(query_lower.split())

        results = []
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            matching_keywords = sum(1 for kw in query_keywords if kw in doc_lower)
            score = matching_keywords / len(query_keywords) if query_keywords else 0

            if score > 0:
                results.append({
                    "document": doc,
                    "id": self.doc_ids[i],
                    "metadata": self.metadatas[i],
                    "similarity": score,
                    "score": score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def retrieve_relevant_context(self, query: str, top_k: int = 3, similarity_threshold: float = 0.1) -> str:
        """Get relevant context as string (ChatService compatible)"""
        results = await self.search(query, top_k=top_k)

        if not results or "results" not in results:
            return ""

        context_parts = []
        for result in results["results"]:
            similarity = result.get("similarity", 0)
            if similarity > similarity_threshold:
                doc = result.get("document", "")
                if doc:
                    context_parts.append(doc)

        return "\n\n".join(context_parts)

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection info"""
        return {
            "name": "rag_collection",
            "initialized": self.initialized,
            "count": len(self.documents),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2" if self.embedding_model else "tfidf_fallback",
            "vector_index": "HNSW" if self.vector_index else "cosine_manual",
            "dimension": self.embedding_dimension
        }

    async def is_ready(self) -> bool:
        """Check if ready"""
        return self.initialized and len(self.documents) > 0

    async def _save_data(self) -> None:
        """Persist data"""
        try:
            data = {
                "documents": self.documents,
                "doc_ids": self.doc_ids,
                "metadatas": self.metadatas,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2" if self.embedding_model else None,
                "vector_index": "HNSW" if self.vector_index else None
            }

            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.data_dir / "documents.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Data saved to {self.data_dir}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
