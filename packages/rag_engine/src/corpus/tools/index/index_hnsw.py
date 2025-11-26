from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd


def build_hnsw(
    base: Path, space: str = "cosine", M: int = 48, ef_construction: int = 200
):
    emb_path = base / "embeddings" / "embeddings.parquet"
    if not emb_path.exists():
        print("Sin embeddings, no se construye HNSW.")
        return
    df = pd.read_parquet(emb_path)
    if len(df) == 0:
        print("Embeddings vacío.")
        return

    idx_dir = base / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_file = idx_dir / "hnsw.idx"
    map_file = idx_dir / "mapping.parquet"

    if idx_file.exists() and map_file.exists():
        mp = pd.read_parquet(map_file)
        known = set(mp["chunk_id"])
        new_df = df[~df["chunk_id"].isin(known)]
        if new_df.empty:
            print("HNSW ya actualizado.")
            return
        dim = len(df["vector"].iloc[0])
        p = hnswlib.Index(space=space, dim=dim)
        p.load_index(str(idx_file))
        start = mp.shape[0]
        mat = np.vstack(new_df["vector"].to_numpy()).astype("float32")
        p.add_items(mat, np.arange(start, start + mat.shape[0]))
        p.save_index(str(idx_file))
        add_map = new_df[["chunk_id", "doc_id", "text", "meta"]]
        pd.concat([mp, add_map], ignore_index=True).to_parquet(map_file, index=False)
        print(f"[OK] HNSW incremental: +{len(new_df)} chunks")
        return

    # full build
    mat = np.vstack(df["vector"].to_numpy()).astype("float32")
    dim = mat.shape[1]
    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=mat.shape[0], ef_construction=ef_construction, M=M)
    p.add_items(mat, np.arange(mat.shape[0]))
    p.save_index(str(idx_file))
    df[["chunk_id", "doc_id", "text", "meta"]].to_parquet(map_file, index=False)
    print("[OK] HNSW listo")
