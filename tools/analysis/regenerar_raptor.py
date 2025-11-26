"""
Regenerador de RAPTOR tree usando embeddings y metadata existentes
Compatible con la base de datos mejorada con contenido comprimido
"""

import json
import logging
import sqlite3
import zlib
from pathlib import Path

import faiss
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAPTOR_Regenerator")


def main():
    # Cargar índice FAISS y metadata
    index_path = Path("data/ml_training/proyecto_COMPLETO_vectordb.index")
    metadata_path = Path("data/ml_training/proyecto_COMPLETO_metadata.json")
    raptor_path = Path("data/ml_training/proyecto_COMPLETO_raptor.json")
    db_path = Path("project_state.db")

    logger.info("📂 Cargando FAISS index y metadata...")
    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        full_meta = json.load(f)

    # Extraer embeddings y metadata de archivos
    num_vectors = index.ntotal
    embeddings = np.zeros((num_vectors, 1024), dtype=np.float32)
    for i in range(num_vectors):
        embeddings[i] = index.reconstruct(int(i))

    metadata = full_meta.get("metadatos_archivos", [])[:num_vectors]

    # Cargar contenido de archivos desde base de datos (si está disponible)
    logger.info("📚 Cargando contenido de archivos desde base de datos...")
    documents = []
    archivos_desde_db = 0
    archivos_desde_disco = 0

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        for meta in metadata:
            archivo = meta.get("archivo", "")
            cursor.execute(
                "SELECT content_compressed FROM files WHERE path = ?", (archivo,)
            )
            result = cursor.fetchone()

            if result and result[0]:
                try:
                    # Descomprimir contenido desde BD
                    decompressed = zlib.decompress(result[0])
                    content = decompressed.decode("utf-8", errors="ignore")[:2000]
                    documents.append(content if content.strip() else archivo)
                    archivos_desde_db += 1
                except Exception:
                    # Fallback: leer desde disco
                    try:
                        with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(2000)
                            documents.append(content if content.strip() else archivo)
                            archivos_desde_disco += 1
                    except Exception:
                        documents.append(archivo)
            else:
                # No está en BD, leer desde disco
                try:
                    with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(2000)
                        documents.append(content if content.strip() else archivo)
                        archivos_desde_disco += 1
                except Exception:
                    documents.append(archivo)

        conn.close()
    else:
        # Sin BD, leer todo desde disco
        logger.warning("⚠️  Base de datos no encontrada, leyendo desde disco...")
        for meta in metadata:
            archivo = meta.get("archivo", "")
            try:
                with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(2000)
                    documents.append(content if content.strip() else archivo)
                    archivos_desde_disco += 1
            except Exception:
                documents.append(archivo)

    logger.info(f"✅ Cargados {num_vectors} vectores")
    logger.info(f"   📦 Desde BD: {archivos_desde_db} archivos")
    logger.info(f"   💾 Desde disco: {archivos_desde_disco} archivos")
    logger.info(f"   📄 Total documentos: {len(documents)}")

    # Crear RAPTOR tree
    logger.info("🌳 Construyendo árbol RAPTOR con lógica corregida...")

    # Level 0: Original documents with file paths
    level_0_metadata = []
    for idx, meta in enumerate(metadata):
        level_0_metadata.append(
            {
                "doc_id": f"L0_D{idx}",
                "archivo": meta.get("archivo", f"unknown_{idx}"),
                "tipo": meta.get("tipo", "unknown"),
                "funciones": meta.get("funciones", 0),
                "clases": meta.get("clases", 0),
            }
        )

    levels = [
        {"docs": documents, "embeddings": embeddings, "metadata": level_0_metadata}
    ]

    # Build 3 levels of hierarchy
    for level in range(1, 4):
        prev_level = levels[-1]

        if len(prev_level["docs"]) < 2:
            logger.info(
                f'⚠️  Nivel {level}: Solo {len(prev_level["docs"])} documentos, terminando jerarquía'
            )
            break

        clusters = []
        cluster_embeddings = []
        cluster_metadata = []

        for i in range(0, len(prev_level["docs"]), 8):
            cluster_docs = prev_level["docs"][i : i + 8]
            cluster_embs = prev_level["embeddings"][i : i + 8]
            cluster_metas = prev_level["metadata"][i : i + 8]

            # Summary
            summary = " | ".join(
                [
                    str(doc)[:100] if len(str(doc)) > 100 else str(doc)
                    for doc in cluster_docs
                ]
            )
            clusters.append(summary)

            # Average embeddings
            avg_emb = np.mean(cluster_embs, axis=0)
            cluster_embeddings.append(avg_emb)

            # Collect children IDs from previous level
            children_ids = []
            for meta in cluster_metas:
                if "doc_id" in meta:
                    children_ids.append(meta["doc_id"])
                elif "cluster_id" in meta:
                    children_ids.append(meta["cluster_id"])

            cluster_meta = {
                "cluster_id": f"L{level}_C{len(clusters)}",
                "num_docs": len(cluster_docs),
                "children": children_ids,
                "total_functions": sum(
                    m.get("funciones", m.get("total_functions", 0))
                    for m in cluster_metas
                ),
                "total_classes": sum(
                    m.get("clases", m.get("total_classes", 0)) for m in cluster_metas
                ),
            }
            cluster_metadata.append(cluster_meta)

        if not clusters:
            break

        logger.info(
            f'  Nivel {level}: {len(clusters)} clusters creados desde {len(prev_level["docs"])} documentos'
        )

        levels.append(
            {
                "docs": clusters,
                "embeddings": np.array(cluster_embeddings),
                "metadata": cluster_metadata,
            }
        )

    raptor_tree = {
        "num_levels": len(levels),
        "level_sizes": [len(level["docs"]) for level in levels],
        "tree": [
            {"level": idx, "metadata": level["metadata"]}
            for idx, level in enumerate(levels)
        ],
    }

    # Guardar
    logger.info(f'💾 Guardando RAPTOR tree con {raptor_tree["num_levels"]} niveles...')
    with open(raptor_path, "w", encoding="utf-8") as f:
        json.dump(raptor_tree, f, indent=2, ensure_ascii=False)

    size_kb = raptor_path.stat().st_size / 1024
    logger.info(f"✅ RAPTOR tree regenerado exitosamente!")
    logger.info(f"📁 Archivo: {raptor_path.name} ({size_kb:.2f} KB)")
    logger.info(f'📊 Estructura de niveles: {raptor_tree["level_sizes"]}')
    logger.info(f'🌳 Total de nodos en el árbol: {sum(raptor_tree["level_sizes"])}')


if __name__ == "__main__":
    main()
