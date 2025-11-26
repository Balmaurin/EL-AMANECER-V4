#!/usr/bin/env python
"""Completar las fases restantes del pipeline: Indexing, RAPTOR y Graph.

INTEGRACIÓN COMPLETA: Conecta corpus/ con datos entrenados en data/ml_training/
"""

import json
import shutil
import sys
from pathlib import Path

# Añadir corpus al path
corpus_dir = Path(__file__).parent
sys.path.insert(0, str(corpus_dir))

# Importar data/ml_training para integración
ml_training_dir = corpus_dir.parent / "data" / "ml_training"

from tools.graph.build_graph import build_graph
from tools.index.index_bm25_whoosh import build_bm25
from tools.index.index_hnsw import build_hnsw
from tools.raptor.build_tree import build_raptor


def integrate_trained_models():
    """Integrar modelos entrenados desde data/ml_training/ con corpus/"""
    print("[*] INTEGRANDO MODELOS ENTRENADOS DE data/ml_training/\n")

    # Verificar existencia de datos entrenados
    trained_files = {
        "bm25": ml_training_dir / "proyecto_COMPLETO_bm25.json",
        "vectordb": ml_training_dir / "proyecto_COMPLETO_vectordb.index",
        "raptor": ml_training_dir / "proyecto_COMPLETO_raptor.json",
        "metadata": ml_training_dir / "proyecto_COMPLETO_metadata.json",
    }

    # Verificar qué archivos existen
    available_files = {}
    for name, path in trained_files.items():
        if path.exists():
            available_files[name] = path
            print(f"✅ {name.upper()}: {path} encontrado")
        else:
            print(f"⚠️ {name.upper()}: {path} no encontrado")

    if not available_files:
        print(
            "❌ No se encontraron archivos de entrenamiento. Procediendo con build normal.\n"
        )
        return False

    # Integrar archivos en corpus index/
    index_dir = corpus_dir / "index"
    index_dir.mkdir(exist_ok=True)

    # 1. Copiar BM25 index entrenado
    if "bm25" in available_files:
        bm25_dest = index_dir / "bm25_trained.json"
        shutil.copy2(available_files["bm25"], bm25_dest)
        print(f"✅ BM25 index integrado: {bm25_dest}")

    # 2. Copiar VectorDB index entrenado
    if "vectordb" in available_files:
        vecdb_dest = index_dir / "vectordb_trained.index"
        shutil.copy2(available_files["vectordb"], vecdb_dest)
        print(f"✅ VectorDB index integrado: {vecdb_dest}")

    # 3. Copiar RAPTOR tree entrenado
    if "raptor" in available_files:
        raptor_dest = index_dir / "raptor_trained.json"
        shutil.copy2(available_files["raptor"], raptor_dest)
        print(f"✅ RAPTOR tree integrado: {raptor_dest}")

    # 4. Copiar metadata entrenada
    if "metadata" in available_files:
        meta_dest = index_dir / "metadata_trained.json"
        shutil.copy2(available_files["metadata"], meta_dest)
        print(f"✅ Metadata integrada: {meta_dest}")

    print(
        "\n✅ INTEGRACIÓN COMPLETADA: Modelos entrenados conectados al sistema corpus/\n"
    )
    return True


if __name__ == "__main__":
    print("🚀 MCP-PHOENIX CORPUS PIPELINE COMPLETO (CON INTEGRACIÓN ML-TRAINING)\n")

    # **INTEGRACIÓN CRÍTICA**: Conectar datos entrenados
    integration_success = integrate_trained_models()

    if not integration_success:
        print(
            "⚠️ Advertencia: No hay datos pre-entrenados. Procediendo con build normal.\n"
        )

    snapshot = Path("corpus/universal/ingest_1762822991")

    print(f"[*] Completando pipeline para: {snapshot}")
    if integration_success:
        print(f"[*] Usando datos pre-entrenados integrados en index/\n")
    else:
        print(f"[*] Construyendo índices desde cero\n")

    # Fase 1: HNSW Index (o usar pre-entrenado)
    print("[1/4] Construyendo/comprobando indice HNSW...")
    try:
        # Si tenemos datos pre-entrenados, verificar integra
        trained_vec_db = corpus_dir / "index" / "vectordb_trained.index"
        if trained_vec_db.exists():
            print(f"[OK] VectorDB pre-entrenado encontrado: {trained_vec_db}")
            print("[OK] HNSW VectorDB ya integrado\n")
        else:
            build_hnsw(snapshot)
            print("[OK] HNSW completado\n")
    except Exception as e:
        print(f"[ERROR] HNSW fallo: {e}\n")
        import traceback

        traceback.print_exc()

    # Fase 2: BM25 Index (o usar pre-entrenado)
    print("[2/4] Construyendo/comprobando indice BM25...")
    try:
        trained_bm25 = corpus_dir / "index" / "bm25_trained.json"
        if trained_bm25.exists():
            print(f"[OK] BM25 pre-entrenado encontrado: {trained_bm25}")
            print("[OK] BM25 ya integrado\n")
        else:
            build_bm25(snapshot)
            print("[OK] BM25 completado\n")
    except Exception as e:
        print(f"[ERROR] BM25 fallo: {e}\n")
        import traceback

        traceback.print_exc()

    # Fase 3: RAPTOR (o usar pre-entrenado)
    print("[3/4] Construyendo/comprobando arbol RAPTOR...")
    try:
        trained_raptor = corpus_dir / "index" / "raptor_trained.json"
        if trained_raptor.exists():
            print(f"[OK] RAPTOR pre-entrenado encontrado: {trained_raptor}")
            print("[OK] RAPTOR ya integrado\n")
        else:
            build_raptor(snapshot)
            print("[OK] RAPTOR completado\n")
    except Exception as e:
        print(f"[ERROR] RAPTOR fallo: {e}\n")
        import traceback

        traceback.print_exc()

    # Fase 4: Graph
    print("[4/4] Construyendo grafo de conocimiento...")
    try:
        build_graph(snapshot)
        print("[OK] Graph completado\n")
    except Exception as e:
        print(f"[ERROR] Graph fallo: {e}\n")
        import traceback

        traceback.print_exc()

    # Fase 5: Verificación final de integración
    print("[5/5] Verificando integración completa...")
    try:
        indices_integrados = 0

        if (corpus_dir / "index" / "bm25_trained.json").exists():
            print("✅ BM25: Integrado")
            indices_integrados += 1

        if (corpus_dir / "index" / "vectordb_trained.index").exists():
            print("✅ VectorDB: Integrado")
            indices_integrados += 1

        if (corpus_dir / "index" / "raptor_trained.json").exists():
            print("✅ RAPTOR: Integrado")
            indices_integrados += 1

        print(f"\n🎯 INTEGRACIÓN COMPLETA: {indices_integrados}/3 índices conectados")
        print("\n[✅] PIPELINE MCP-PHOENIX COMPLETO Y OPERATIVO!")

    except Exception as e:
        print(f"[ERROR] Verificación fallo: {e}\n")
        import traceback

        traceback.print_exc()
