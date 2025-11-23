#!/usr/bin/env python3
"""
RAG REAL - Sistema Vectorial TF-IDF + SVD Funcional
===================================================

Sistema completamente real sin simulaciones:
- TF-IDF vectorization real con sklearn
- SVD dimensionality reduction real
- Cosine similarity real
- Persistence real
"""

import asyncio
import hashlib
import json
import pickle
from pathlib import Path

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    raise ImportError("Requiere sklearn. Ejecuta: pip install scikit-learn")


class RealRAGService:
    """
    RAG Service usando TF-IDF + SVD con compatibilidad ChatService
    """

    def __init__(self, data_dir="./rag_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.documents = []
        self.doc_ids = []
        self.metadatas = []
        self.vectorizer = None
        self.svd = None
        self.tfidf_matrix = None
        self.reduced_matrix = None

        self._load_data()
        print(f"RAG REAL inicializado - {len(self.documents)} documentos")

    # ========== MÉTODOS DE COMPATIBILIDAD CON CHATSERVICE ==========

    def get_collection_info(self):
        """Método de compatibilidad con ChatService"""
        return {
            "initialized": self.vectorizer is not None,
            "count": len(self.documents),
            "name": "tfidf_svd_collection",
            "method": "tfidf_svd_cosine",
            "dimensions": self.reduced_matrix.shape[1] if self.reduced_matrix is not None else 0,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        }

    async def is_ready(self):
        """Método de compatibilidad con ChatService"""
        return len(self.documents) > 0 and self.vectorizer is not None

    async def retrieve_relevant_context(self, query: str, top_k: int = 3, similarity_threshold: float = 0.1):
        """Método de compatibilidad - alias de search formateado como string"""
        results = await self.search(query, top_k=top_k)

        if not results or "results" not in results:
            return ""

        # Formatear como contexto string
        context_parts = []
        for result in results["results"]:
            similarity = result.get("similarity", 0)
            if similarity > similarity_threshold:
                doc = result.get("document", "")
                if doc:
                    context_parts.append(doc)

        return "\n\n".join(context_parts)

    # ========== MÉTODOS CORE DEL RAG SYSTEM ==========

    def _persist_embeddings(self):
        """Persistencia adicional de artefactos de embeddings para health checker."""
        try:
            if not self.vectorizer or not self.svd:
                return
            embed_dir = Path("data/embeddings")
            embed_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "type": "tfidf_svd",
                "vocab_size": len(self.vectorizer.vocabulary_),
                "vocabulary": list(self.vectorizer.vocabulary_.keys())[:2000],
                "idf": getattr(self.vectorizer, "idf_", [])[:2000],
                "n_features": self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else None,
                "svd_components": self.svd.components_.tolist() if hasattr(self.svd, "components_") else [],
                "explained_variance_ratio": getattr(self.svd, "explained_variance_ratio_", []).tolist() if hasattr(self.svd, "explained_variance_ratio_") else [],
                "n_components": getattr(self.svd, "n_components", None) or (self.svd.components_.shape[0] if hasattr(self.svd, "components_") else None),
                "doc_count": len(self.documents),
            }
            out_file = embed_dir / "rag_tfidf_svd.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            # Silencioso: no romper indexado si falla persistencia opcional
            pass

    def _load_data(self):
        """Cargar datos persistidos"""
        try:
            data_file = self.data_dir / "data.json"
            if data_file.exists():
                with open(data_file, "r") as f:
                    data = json.load(f)
                    self.documents = data["documents"]
                    self.doc_ids = data["doc_ids"]
                    self.metadatas = data["metadatas"]
        except:
            pass

    def _save_data(self):
        """Guardar datos"""
        data = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "metadatas": self.metadatas,
        }
        with open(self.data_dir / "data.json", "w") as f:
            json.dump(data, f)

    async def index_documents(self, docs, ids=None, metadatas=None):
        """Indexar documentos con TF-IDF real + SVD"""
        if not docs:
            return {"error": "no documents"}

        if not ids:
            ids = [f"doc_{i}" for i in range(len(docs))]
        if not metadatas:
            metadatas = [{}] * len(docs)

        self.documents = docs
        self.doc_ids = ids
        self.metadatas = metadatas

        # TF-IDF vectorization (soporta español y otros idiomas)
        self.vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)

        # SVD reduction - ajustar componentes a features disponibles
        n_features = self.tfidf_matrix.shape[1]

        # Solo aplicar SVD si hay suficientes features (más de 2)
        if n_features > 2:
            n_components = min(50, n_features - 1)  # máximo 50 o n_features-1
            self.svd = TruncatedSVD(n_components=n_components)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                self.reduced_matrix = self.svd.fit_transform(self.tfidf_matrix)
        else:
            # Si hay pocas features, usar TF-IDF directamente sin reducción
            self.svd = None
            self.reduced_matrix = self.tfidf_matrix.toarray()
            n_components = n_features

        self._save_data()
        # Persistir artefactos de embeddings para health checker
        self._persist_embeddings()
        return {"indexed": len(docs), "dimensions": n_components}

    async def search(self, query, top_k=3):
        """Búsqueda vectorial real"""
        if not self.vectorizer:
            return {"error": "no index built"}

        query_vec = self.vectorizer.transform([query])

        # Si hay SVD, aplicarlo; si no, usar TF-IDF directo
        if self.svd is not None:
            query_reduced = self.svd.transform(query_vec)
        else:
            query_reduced = query_vec.toarray()

        similarities = cosine_similarity(query_reduced, self.reduced_matrix)[0]
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "document": self.documents[idx][:200] + "...",
                    "id": self.doc_ids[idx],
                    "metadata": self.metadatas[idx],
                    "similarity": round(float(similarities[idx]), 4),
                }
            )

        return {"query": query, "results": results, "method": "tfidf_svd_cosine"}


# DEMO
async def demo():
    print("🔬 RAG REAL - Sistema Vectorial Funcional")
    print("=" * 50)

    rag = RealRAGService()

    docs = [
        "Sheily AI is an advanced AI platform using machine learning",
        "RAG architecture combines information retrieval with text generation",
        "Vector embeddings represent concepts as mathematical vectors",
        "Natural language processing enables human-like understanding",
    ]

    print("📚 Indexando documentos con TF-IDF real...")
    result = await rag.index_documents(docs)
    print(f"✅ Indexación: {result['indexed']} documentos")

    print("\n🔍 Búsquedas vectoriales reales:")
    queries = ["What is AI?", "how does RAG work?", "explain embeddings"]

    for query in queries:
        results = await rag.search(query)
        print(f"\nQuery: '{query}'")
        for i, r in enumerate(results["results"][:1], 1):
            print(f"  {i}. Score: {r['similarity']:.4f} - '{r['document'][:50]}...'")


if __name__ == "__main__":
    asyncio.run(demo())
