import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.common.paths import INDEX_DIR

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

import hnswlib  # type: ignore
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger("rag.build_hnsw")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )


def load_embeddings_from_faiss(index_path: Path) -> np.ndarray:
    if faiss is None:
        raise RuntimeError(
            "FAISS no disponible para reconstruir embeddings del índice."
        )
    index = faiss.read_index(str(index_path))
    n = index.ntotal
    # Se asume IndexFlat*, reconstruct está disponible
    vec0 = index.reconstruct(0)
    dim = vec0.shape[0]
    mat = np.empty((n, dim), dtype=np.float32)
    mat[0] = vec0
    for i in range(1, n):
        mat[i] = index.reconstruct(i)
    return mat


def encode_embeddings_from_mapping(mapping_path: Path, model_name: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Cargando modelo de embeddings: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device)
    df = pd.read_parquet(mapping_path, engine="pyarrow")
    texts = df["text"].astype(str).tolist()
    embeds = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(
        np.float32
    )
    return embeds


def build_hnsw_from_embeddings(
    embeds: np.ndarray,
    out_path: Path,
    space: str = "cosine",
    M: int = 16,
    ef_construction: int = 200,
):
    n, dim = embeds.shape
    # normalizar si se usa 'cosine'
    if space == "cosine":
        norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12
        embeds = embeds / norms

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(embeds, np.arange(n))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(out_path))
    log.info(f"Índice HNSW guardado en: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Construye un índice HNSW (hnswlib) para el corpus"
    )
    ap.add_argument(
        "--index-dir",
        default=None,
        help="Carpeta de índices (por defecto: proyecto/index/)",
    )
    ap.add_argument(
        "--mapping",
        default="corpus_mapping.parquet",
        help="Nombre de mapping parquet dentro de index-dir",
    )
    ap.add_argument(
        "--faiss",
        default="corpus_faiss.index",
        help="Nombre de índice FAISS dentro de index-dir",
    )
    ap.add_argument(
        "--out",
        default="hnsw.idx",
        help="Nombre de índice HNSW de salida dentro de index-dir",
    )
    ap.add_argument(
        "--model", default="BAAI/bge-m3", help="Modelo de embeddings si no hay FAISS"
    )
    ap.add_argument(
        "--reencode", action="store_true", help="Forzar re-embebido desde mapping"
    )
    args = ap.parse_args()

    # Use absolute INDEX_DIR or override from --index-dir
    index_dir = Path(args.index_dir) if args.index_dir else INDEX_DIR
    index_dir = index_dir.resolve()

    mapping_path = index_dir / args.mapping
    faiss_path = index_dir / args.faiss
    out_path = index_dir / args.out

    if not mapping_path.exists():
        raise FileNotFoundError(f"No existe mapping: {mapping_path}")

    if not args.reencode and faiss is not None and faiss_path.exists():
        log.info(f"Reconstruyendo embeddings desde FAISS: {faiss_path}")
        embeds = load_embeddings_from_faiss(faiss_path)
    else:
        log.info("Generando embeddings desde mapping con el modelo.")
        embeds = encode_embeddings_from_mapping(mapping_path, args.model)

    build_hnsw_from_embeddings(embeds, out_path)


if __name__ == "__main__":
    main()
