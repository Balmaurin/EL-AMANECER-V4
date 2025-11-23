from pathlib import Path
from typing import Any, Dict, List, TypeVar, Union

import numpy as np
import pandas as pd

T = TypeVar("T", bound="FaissSearcher")

# Importación condicional de faiss
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None  # type: ignore


class FaissSearcher:
    """Clase para realizar búsquedas con FAISS."""

    def __init__(self, base_path: Path) -> None:
        """
        Inicializa el buscador FAISS.

        Args:
            base_path: Directorio base del índice

        Raises:
            ImportError: Si FAISS no está instalado
            RuntimeError: Si los archivos del índice no existen
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS no está instalado. Instale con: pip install faiss-cpu (o faiss-gpu)"
            )

        index_path = base_path / "index" / "faiss.ivfpq"
        mapping_path = base_path / "index" / "mapping.parquet"

        if not index_path.exists():
            raise RuntimeError(f"No se encontró el índice FAISS en {index_path}")

        if not mapping_path.exists():
            raise RuntimeError(f"No se encontró el mapeo de chunks en {mapping_path}")

        self.index = faiss.read_index(str(index_path))  # type: ignore
        self.mapping = pd.read_parquet(mapping_path)

    def search(
        self, query_vector: np.ndarray, limit: int = 10
    ) -> List[Dict[str, Union[str, float, int, dict]]]:
        """
        Realiza una búsqueda por similitud usando FAISS.

        Args:
            query_vector: Vector de consulta normalizado
            limit: Número máximo de resultados

        Returns:
            Lista de documentos encontrados con sus scores

        Raises:
            RuntimeError: Si hay problemas con la búsqueda
        """
        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, limit)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS devuelve -1 para resultados inválidos
                continue

            chunk = self.mapping.iloc[idx]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "text": chunk["text"],
                    "meta": chunk["meta"],
                    "score": float(score),
                    "rank": i + 1,
                }
            )

        return results


def build_faiss(branch: str, base: Path) -> None:
    """
    Construye un índice FAISS.

    Args:
        branch: Nombre del branch de datos
        base: Directorio base

    Raises:
        ImportError: Si FAISS no está instalado
        RuntimeError: Si hay problemas construyendo el índice
    """
    # Import aquí para no requerir faiss en toda la aplicación
    if not hasattr(build_faiss, "_faiss"):
        try:
            import faiss

            build_faiss._faiss = faiss
        except ImportError as e:
            raise ImportError(
                "FAISS no disponible. Instale con: pip install faiss-cpu (o faiss-gpu)"
            ) from e
    faiss = build_faiss._faiss
    emb_path = base / "embeddings" / "embeddings.parquet"
    if not emb_path.exists():
        print("Sin embeddings, no se construye FAISS.")
        return
    df = pd.read_parquet(emb_path)
    if len(df) == 0:
        print("Embeddings vacío.")
        return
    mat = np.vstack(df["vector"].to_numpy()).astype("float32")
    dim = mat.shape[1]
    # Index IVF+PQ por defecto
    nlist = min(4096, max(64, int(len(df) / 100)))
    quant = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quant, dim, nlist, 16, 8)
    faiss.normalize_L2(mat)
    index.train(mat)
    index.add(mat)
    out_dir = base / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.ivfpq"))
    # mapa de chunks
    df[["chunk_id", "doc_id", "text", "meta"]].to_parquet(
        out_dir / "mapping.parquet", index=False
    )
    print("[OK] FAISS listo")
