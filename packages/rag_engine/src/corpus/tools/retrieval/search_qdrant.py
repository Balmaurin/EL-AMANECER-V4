from pathlib import Path
from typing import Dict, List

import yaml


def search_qdrant(base: Path, query: str, top_k: int = 10) -> List[Dict]:
    try:
        from qdrant_client import QdrantClient
    except Exception as e:
        print("qdrant-client no disponible:", e)
        return []
    from sentence_transformers import SentenceTransformer

    cfg = yaml.safe_load(open("config/universal.yaml", "r", encoding="utf-8"))
    qcfg = cfg.get("external_index", {}).get("qdrant", {})
    coll = qcfg.get("collection", "rag_universal")
    client = QdrantClient(
        host=qcfg.get("host", "127.0.0.1"), port=int(qcfg.get("port", 6333))
    )
    model = SentenceTransformer(cfg.get("embedder", {}).get("model", "BAAI/bge-m3"))
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[
        0
    ].tolist()
    res = client.search(
        collection_name=coll, query_vector=vec, limit=top_k, with_payload=True
    )
    out = []
    for r in res:
        p = r.payload or {}
        out.append(
            {
                "chunk_id": p.get("chunk_id", ""),
                "doc_id": p.get("doc_id", ""),
                "title": p.get("title", ""),
                "text": p.get("text", ""),
                "score": float(r.score),
                "meta": p.get("meta", {}),
            }
        )
    return out
