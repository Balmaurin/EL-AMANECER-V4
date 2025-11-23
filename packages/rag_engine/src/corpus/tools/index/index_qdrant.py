from pathlib import Path
from typing import Optional

import pandas as pd


def upsert_qdrant(
    base: Path,
    collection: str = "rag_universal",
    host: str = "127.0.0.1",
    port: int = 6333,
):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams
    except Exception as e:
        print("qdrant-client no disponible:", e)
        return
    emb = base / "embeddings" / "embeddings.parquet"
    if not emb.exists():
        print("No hay embeddings para subir a Qdrant")
        return
    df = pd.read_parquet(emb)
    if len(df) == 0:
        print("Embeddings vacío")
        return
    # vector dimension
    dim = len(df["vector"].iloc[0])
    client = QdrantClient(host=host, port=port)
    # ensure collection
    try:
        client.get_collection(collection)
    except Exception:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    # payloads and points
    points = []
    for i, row in df.iterrows():
        payload = {
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "text": row["text"],
        }
        try:
            m = row.get("meta", {})
            if isinstance(m, str):
                import json

                m = json.loads(m)
            payload["title"] = m.get("title", "")
            payload["meta"] = m
        except Exception:
            payload["title"] = ""
        points.append((row["chunk_id"], row["vector"], payload))
    # upsert in batches
    B = 1024
    for i in range(0, len(points), B):
        batch = points[i : i + B]
        ids = [p[0] for p in batch]
        vecs = [p[1] for p in batch]
        payloads = [p[2] for p in batch]
        client.upsert(
            collection_name=collection,
            points=client._construct_points(ids=ids, vectors=vecs, payloads=payloads),
        )
    print(f"[OK] Qdrant upsert: {len(points)} points -> {collection}")
