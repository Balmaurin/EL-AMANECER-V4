from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def search_milvus(base: Path, query: str, top_k: int = 10) -> List[Dict]:
    try:
        from pymilvus import Collection, connections
    except Exception as e:
        print("pymilvus no disponible:", e)
        return []
    from sentence_transformers import SentenceTransformer

    cfg = yaml.safe_load(open("config/universal.yaml", "r", encoding="utf-8"))
    mcfg = cfg.get("external_index", {}).get("milvus", {})
    coll_name = mcfg.get("collection", "rag_universal")
    connections.connect(
        alias="default",
        host=mcfg.get("host", "127.0.0.1"),
        port=int(mcfg.get("port", 19530)),
        user=mcfg.get("user", "root"),
        password=mcfg.get("password", ""),
        secure=bool(mcfg.get("secure", False)),
    )
    coll = Collection(coll_name)
    coll.load()
    model = SentenceTransformer(cfg.get("embedder", {}).get("model", "BAAI/bge-m3"))
    vec = (
        model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        .astype("float32")
        .tolist()
    )
    r = coll.search(
        data=vec,
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["chunk_id", "doc_id", "title", "text"],
    )
    out = []
    for hits in r:
        for h in hits:
            ent = h.entity
            out.append(
                {
                    "chunk_id": ent.get("chunk_id"),
                    "doc_id": ent.get("doc_id"),
                    "title": ent.get("title"),
                    "text": ent.get("text"),
                    "score": float(h.distance),
                }
            )
    return out
