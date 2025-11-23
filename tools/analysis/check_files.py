import json

import faiss

# Cargar metadata
with open(
    "data/ml_training/proyecto_COMPLETO_metadata.json", "r", encoding="utf-8"
) as f:
    metadata = json.load(f)

# Cargar índice FAISS
index = faiss.read_index("data/ml_training/proyecto_COMPLETO_vectordb.index")

print("📊 DISCREPANCIA DE ARCHIVOS:")
print("=" * 60)
print(f"Archivos en metadata JSON    : {len(metadata['metadatos_archivos'])}")
print(f"Vectores en índice FAISS     : {index.ntotal}")
print(f"Total archivos (metadata)    : {metadata['total_archivos']}")
print(f"Archivos procesados (metadata): {metadata['archivos_procesados']}")
print()
print(f"❌ DIFERENCIA: {len(metadata['metadatos_archivos']) - index.ntotal} archivos")
print()

if len(metadata["metadatos_archivos"]) > index.ntotal:
    print("⚠️  Hay archivos en metadata que NO tienen vectores en FAISS")
    print(
        "    Esto significa que algunos archivos se analizaron pero no se vectorizaron"
    )
