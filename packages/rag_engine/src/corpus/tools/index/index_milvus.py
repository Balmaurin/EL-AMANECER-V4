import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def upsert_milvus(
    base: Path,
    collection: str = "rag_universal",
    host: str = "127.0.0.1",
    port: int = 19530,
    user: str = "root",
    password: str = None,  # Will use MILVUS_PASSWORD env var if None
    secure: bool = False,
):
    if password is None:
        password = os.getenv("MILVUS_PASSWORD", "")
    try:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )
    except Exception as e:
        print("pymilvus no disponible:", e)
        return
    emb = base / "embeddings" / "embeddings.parquet"
    if not emb.exists():
        print("No hay embeddings para subir a Milvus")
        return
    df = pd.read_parquet(emb)
    if len(df) == 0:
        print("Embeddings vacío")
        return
    dim = len(df["vector"].iloc[0])
    connections.connect(
        alias="default",
        host=host,
        port=port,
        user=user,
        password=password,
        secure=secure,
    )
    # schema
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="RAG universal collection")
    if utility.has_collection(collection):
        coll = Collection(collection)
    else:
        coll = Collection(collection, schema=schema)
    # insert
    data = [
        None,  # pk auto
        df["chunk_id"].tolist(),
        df["doc_id"].tolist(),
        [
            (
                row.get("meta", {}).get("title", "")
                if isinstance(row.get("meta", {}), dict)
                else ""
            )
            for _, row in df.iterrows()
        ],
        df["text"].tolist(),
        np.vstack(df["vector"].to_numpy()).astype("float32").tolist(),
    ]
    coll.insert(data)
    # index + load
    try:
        coll.create_index(
            field_name="vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 48, "efConstruction": 200},
            },
        )
    except Exception as e:
        logger.debug(f"Index already exists or creation failed: {e}")
    coll.load()
    print(f"[OK] Milvus upsert: {len(df)} vectors -> {collection}")
