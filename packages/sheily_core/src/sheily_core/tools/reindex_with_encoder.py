#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHEILY REINDEXER ZERO-DEPENDENCY
Script para reindexar embeddings sin dependencias externas

DEPENDENCIAS: Solo Python stdlib
ALTERNATIVA: Para embeddings reales, usar rag_zero_deps.py con TF-IDF
"""
import argparse
import json
import logging
import math
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def create_simple_embeddings(texts, dims=384):
    """
    Crea embeddings básicos usando hash y características del texto
    No es tan avanzado como sentence-transformers, pero funciona sin dependencias
    """
    embeddings = []

    for text in texts:
        # Vector base usando hash del texto
        text_hash = hash(text.lower()) % (2**31)

        # Características del texto
        features = {
            "length": len(text),
            "words": len(text.split()),
            "chars": len(set(text.lower())),
            "digits": sum(c.isdigit() for c in text),
            "upper": sum(c.isupper() for c in text),
            "spaces": text.count(" "),
            "punctuation": sum(c in ".,!?;:" for c in text),
        }

        # Crear vector de dimensión fija
        vector = []
        for i in range(dims):
            # Combinar hash con características
            seed = (text_hash + i) % (2**31)

            # Normalizar características
            feature_sum = sum(features.values())
            norm_features = feature_sum / (1 + i % 10)

            # Valor del vector
            value = math.sin(seed + norm_features) * math.cos(i)
            vector.append(value)

        # Normalizar vector (simulando normalize_embeddings=True)
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        embeddings.append(vector)

    return embeddings


def save_embeddings_sqlite(vectors_path, embeddings, ids):
    """Guardar embeddings en SQLite en lugar de numpy"""
    db_path = str(vectors_path).replace(".npz", ".db")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Crear tabla
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            vector TEXT,
            dimension INTEGER
        )
    """
    )

    # Insertar embeddings
    for i, (embedding, doc_id) in enumerate(zip(embeddings, ids)):
        vector_json = json.dumps(embedding)
        cursor.execute(
            "INSERT OR REPLACE INTO embeddings (id, vector, dimension) VALUES (?, ?, ?)",
            (str(doc_id), vector_json, len(embedding)),
        )

    conn.commit()
    conn.close()

    logger.info(f"✅ Embeddings guardados en SQLite: {db_path}")
    return db_path


def main():
    logger.info("🔄 SHEILY REINDEXER ZERO-DEPENDENCY")
    logger.info("=" * 50)
    ap = argparse.ArgumentParser(
        description="Reindexar con embeddings básicos (zero-dependency)"
    )
    ap.add_argument("--run_dir", required=True, help="branches/<rama>/<fecha>")
    ap.add_argument(
        "--encoder_path",
        default="basic_hash",
        help="Tipo de encoder (basic_hash, tfidf)",
    )
    ap.add_argument(
        "--dimensions", type=int, default=384, help="Dimensiones del vector"
    )
    args = ap.parse_args()

    run = Path(args.run_dir)
    texts_path = run / "st" / "texts.jsonl"
    vectors_path = run / "st" / "vectors.npz"
    meta_path = run / "st" / "meta.json"
    index_path = run / "st" / "index.json"

    if not texts_path.exists():
        logger.error(f"❌ Falta {texts_path}")
        return

    # Leer textos
    ids, texts = [], []
    with texts_path.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            ids.append(j["id"])
            texts.append(j["text"])

    logger.info(f"📄 Procesando {len(texts)} documentos")
    logger.info(f"🔧 Encoder: {args.encoder_path} (dimensiones: {args.dimensions})")
    # Crear embeddings básicos
    if args.encoder_path == "tfidf":
        # Usar el sistema TF-IDF de rag_zero_deps.py
        logger.warning(
            "⚠️  Para embeddings TF-IDF avanzados, usar: python scripts/rag_zero_deps.py"
        )
        logger.info("📝 Usando embeddings básicos por hash...")
    embeddings = create_simple_embeddings(texts, args.dimensions)

    # Guardar embeddings en SQLite
    db_path = save_embeddings_sqlite(vectors_path, embeddings, ids)

    # Crear metadata
    meta = {
        "mode": "zero_deps",
        "vector_model": f"basic_embeddings_{args.encoder_path}",
        "dims": args.dimensions,
        "total_chunks": len(embeddings),
        "method": "hash_based_features",
        "database": db_path,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    # Crear índice si no existe
    if not index_path.exists():
        index_data = {
            "ids": ids,
            "method": "zero_dependency",
            "embedding_type": "basic_hash",
        }
        index_path.write_text(
            json.dumps(index_data, ensure_ascii=False, indent=2), "utf-8"
        )

    logger.info(f"✅ Reindex completado:")
    logger.info(f"   📊 Metadata: {meta_path}")
    logger.info(f"   🗃️  Base datos: {db_path}")
    logger.info(f"   📑 Índice: {index_path}")
    logger.info("\n💡 Para búsqueda semántica avanzada, usar:")
    logger.info(
        "   python scripts/rag_zero_deps.py --action search --query 'tu consulta'"
    )


if __name__ == "__main__":
    main()
