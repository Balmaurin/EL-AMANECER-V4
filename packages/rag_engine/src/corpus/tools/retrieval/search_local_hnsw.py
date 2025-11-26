"""
Local HNSW/FAISS vector search that supports provider-based query encoding.

If config embedder.provider == 'openai', queries are encoded using OpenAI
embeddings API, avoiding local Torch entirely. Otherwise falls back to
SentenceTransformers (may require Torch).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from tools.common.utils import safe_query

# Windows stability: pre-import torch as early as possible to avoid DLL init issues
try:
    import torch  # type: ignore  # noqa: F401
except Exception:
    # Best-effort: dense search will still try to import it later if needed
    pass
# Cache is optional; import lazily to avoid hard dependency on Redis
try:
    from tools.retrieval.cache_manager import get_cache  # type: ignore
except Exception:
    get_cache = None  # type: ignore

log = logging.getLogger("rag.retrieval.hnsw")


def _load_provider_and_model() -> tuple[str, str]:
    try:
        import yaml

        cfg = yaml.safe_load(open("config/universal.yaml", "r", encoding="utf-8"))
        emb = cfg.get("embedder", {})
        provider = str(emb.get("provider", "local")).lower()
        model = str(emb.get("model", "BAAI/bge-m3"))
        return provider, model
    except Exception as e:
        log.warning(f"Failed to load provider/model from config: {e}")
        return "local", "BAAI/bge-m3"


class HNSWSearcher:
    """HNSW/FAISS hybrid vector searcher with provider-based encoding."""

    def __init__(
        self, index_path: Path, mapping_path: Path, provider: str, model_name: str
    ):
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index file: {index_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing mapping file: {mapping_path}")
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.provider = provider
        self.model_name = model_name
        # Prepare query encoder according to provider
        if self.provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                log.error(f"OpenAI SDK not installed: {e}")
                raise
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set; cannot run dense search with OpenAI provider"
                )
            self._client = OpenAI(api_key=api_key)
            self._encode = self._encode_openai
        else:
            # Fallback to SentenceTransformers
            # Harden Windows import order to avoid DLL init issues (WinError 1114)
            try:
                # Force CPU and disable CUDA device discovery to prevent accidental GPU DLL loads
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
                # Pre-import torch BEFORE sentence_transformers to stabilize DLL loading on Windows
                import torch  # type: ignore

                _ = torch.__version__  # touch to ensure import
            except Exception as e:
                log.debug(f"Torch pre-import skipped/failed: {e}")
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as e:
                log.exception("sentence_transformers import failed")
                raise RuntimeError(
                    "SentenceTransformers not available for local provider"
                ) from e
            # Always use CPU here for stability on Windows; model uses small BGE by default
            device = "cpu"
            try:
                self._st_model = SentenceTransformer(model_name, device=device)
            except OSError as e:
                # Retry once after clearing possible torch caches/env
                log.warning(f"Retrying ST load on CPU due to OS error: {e}")
                try:
                    import torch  # type: ignore

                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                self._st_model = SentenceTransformer(model_name, device=device)
            self._encode = self._encode_st
        self.mapping = pd.read_parquet(mapping_path, engine="pyarrow").reset_index(
            drop=True
        )
        self.backend = None
        self._load_index()

    def _load_index(self):
        dim = self._encode_query("probe").shape[1]
        # Lazy import backends
        try:
            import hnswlib  # type: ignore
        except Exception:
            hnswlib = None  # type: ignore
        try:
            import faiss  # type: ignore
        except Exception:
            faiss = None  # type: ignore

        # Decide backend based on file name first (safer if both are installed)
        name = str(self.index_path.name).lower()
        want_backend = None
        if "faiss" in name or name.endswith(".faiss") or name.endswith(".index"):
            want_backend = "faiss"
        if "hnsw" in name or name.endswith(".idx") or name.endswith(".hnsw"):
            want_backend = "hnswlib"

        # Try desired backend, then fallback to the other one
        last_err = None
        for backend in ([want_backend] if want_backend else []) + [
            b for b in ["faiss", "hnswlib"] if b != want_backend
        ]:
            try:
                if backend == "hnswlib" and hnswlib is not None:
                    self.index = hnswlib.Index(space="cosine", dim=dim)
                    self.index.load_index(str(self.index_path))
                    self.backend = "hnswlib"
                    return
                if backend == "faiss" and faiss is not None:
                    self.index = faiss.read_index(str(self.index_path))
                    self.backend = "faiss"
                    return
            except Exception as e:  # keep trying
                last_err = e

        if hnswlib is None and faiss is None:
            raise RuntimeError("No vector backend (hnswlib/faiss) available")
        # If we reach here, a backend was present but failed to load the file
        raise RuntimeError(
            f"Failed to load index {self.index_path} with available backends: {last_err}"
        )

    def _encode_st(self, texts: List[str]) -> np.ndarray:
        return self._st_model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        out = []
        B = 2048
        for i in range(0, len(texts), B):
            batch = texts[i : i + B]
            resp = self._client.embeddings.create(model=self.model_name, input=batch)
            out.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
        return np.vstack(out)

    def _encode_query(self, query: str) -> np.ndarray:
        return self._encode([safe_query(query)])

    def _process_metadata(self, meta):
        if isinstance(meta, str):
            try:
                return json.loads(meta)
            except Exception:
                return {}
        return meta if isinstance(meta, dict) else {}

    def search(
        self, query: str, top_k: int = 10, ef_search=None, use_cache: bool = True
    ):
        if not query:
            return []

        # Check cache first
        cache = get_cache() if get_cache else None
        if use_cache and cache and cache.health_check():
            cached_results = cache.get(query, top_k)
            if cached_results is not None:
                log.debug(f"[+] Cache hit for query: {query[:30]}...")
                return cached_results

        qv = self._encode_query(query)
        results = []
        try:
            if self.backend == "hnswlib":
                ef_search = ef_search or max(50, top_k * 4)
                self.index.set_ef(ef_search)
                labels, distances = self.index.knn_query(qv, k=max(1, top_k))
                # HNSW cosine distance is in [0, 2]; convert to similarity [0, 1]
                # For unit vectors: similarity = 1 - distance/2
                scores = np.clip(1.0 - (distances[0] / 2.0), 0.0, 1.0)
            elif self.backend == "faiss":
                distances, labels = self.index.search(qv, k=max(1, top_k))
                # FAISS L2 distance: convert to similarity; clip to [0, 1]
                scores = np.clip(1.0 - np.sqrt(np.maximum(distances[0], 0.0)), 0.0, 1.0)
            else:
                raise RuntimeError("No valid backend")
            # Keep scores as-is; no min-max normalization (preserve query-dependent variation)
            for idx, sc in zip(labels[0], scores):
                row = self.mapping.iloc[int(idx)]
                meta = self._process_metadata(row.get("meta", {}))
                results.append(
                    {
                        "chunk_id": row.get("chunk_id", str(idx)),
                        "doc_id": row.get("doc_id", ""),
                        "title": meta.get("title", ""),
                        "text": row.get("text", ""),
                        "score": float(sc),
                        "source": "dense",
                        "meta": meta,
                    }
                )
        except Exception as e:
            log.error(f"Search failed: {e}")

        # Cache results if search was successful
        if results and use_cache and get_cache:
            try:
                cache = get_cache()
                if cache and cache.health_check():
                    cache.set(query, top_k, results, ttl=3600)  # 1 hour TTL
            except Exception as e:
                log.warning(f"Failed to cache results: {e}")

        return results


def search_local_hnsw(base: Path, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Execute HNSW search over vector index.

    Args:
        base: Base directory containing indices
        query: Search query string
        top_k: Number of results to return

    Returns:
        List of search results sorted by similarity
    """
    # Use snapshot index directory and mapping path
    index_dir = base / "index"
    mapping_path = index_dir / "mapping.parquet"
    index_candidates = [
        index_dir / "hnsw.idx",
        index_dir / "corpus_faiss.index",
        index_dir / "index.faiss",
        index_dir / "faiss.index",
    ]
    index_path = None
    for candidate in index_candidates:
        if candidate.exists():
            index_path = candidate
            break

    if index_path is None or not mapping_path.exists():
        log.error(
            "Required index files not found (tried: hnsw.idx, faiss.index, index.faiss)"
        )
        return []

    try:
        # Initialize and execute search
        # Load config to pick provider and model
        try:
            cfg = yaml.safe_load(open("config/universal.yaml", "r", encoding="utf-8"))
        except Exception:
            cfg = {}
        embed_cfg = cfg.get("embedder", {})
        provider = embed_cfg.get("provider", "local")
        model = embed_cfg.get("model", "BAAI/bge-m3")
        searcher = HNSWSearcher(
            index_path=index_path,
            mapping_path=mapping_path,
            provider=provider,
            model_name=model,
        )
        return searcher.search(query, top_k=top_k)
    except Exception as e:
        log.error(f"Search failed: {e}")
        return []
