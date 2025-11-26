import os

# Pre-import torch to stabilize DLL loading on Windows
try:  # noqa: SIM105
    import torch  # type: ignore  # noqa: F401
except Exception:
    pass
from typing import Any, Dict, List, Optional, Union

from rapidfuzz import fuzz


class LexicalReranker:
    name = "lexical-fallback"

    def rerank(
        self, query: str, hits: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        if not hits:
            return hits
        for h in hits:
            text = h.get("text", "")
            h["score"] = 0.6 * (
                fuzz.token_set_ratio(query, text) / 100.0
            ) + 0.4 * h.get("score", 0.0)
        return sorted(hits, key=lambda x: -x["score"])[:top_k]


class LocalReranker:
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None
    ):
        try:
            from sentence_transformers import CrossEncoder

            self.m = CrossEncoder(model_name, device=(device or "cpu"))
            self.ok = True
            self.name = f"cross-encoder:{model_name}"
        except Exception:
            self.m = None
            self.ok = False
            self.error = f"Failed to load reranker model '{model_name}'"
            self.name = "lexical-fallback"

    def rerank(
        self, query: str, hits: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        if (not hits) or (not self.ok):
            return hits
        pairs = [[query, h.get("text", "")] for h in hits]
        scores = self.m.predict(pairs).tolist()
        for h, s in zip(hits, scores):
            h["score"] = float(s)
        return sorted(hits, key=lambda x: -x["score"])[:top_k]


def build_reranker(
    *, enabled: bool = True, model: Optional[str] = None, device_hint: str = "cpu"
) -> Union[LexicalReranker, LocalReranker]:
    env_enabled = os.getenv("RAG_retrieval__rerank__enabled", "true").lower() == "true"

    if not (enabled and env_enabled):
        return LexicalReranker()

    model_name = model or os.getenv(
        "RAG_retrieval__rerank__model", "BAAI/bge-reranker-base"
    )

    device = device_hint
    if isinstance(device_hint, str):
        hint = device_hint.strip().lower()
        if hint in {"", "auto", "none", "default"}:
            device = None

    reranker = LocalReranker(model_name=model_name, device=device)
    if getattr(reranker, "ok", False):
        return reranker

    fallback = LexicalReranker()
    if hasattr(reranker, "error"):
        setattr(fallback, "error", reranker.error)
    return fallback
