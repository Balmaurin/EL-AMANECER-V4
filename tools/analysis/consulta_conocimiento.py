"""
🎯 SISTEMA DE CONSULTA INTELIGENTE
Consulta el conocimiento completo del proyecto entrenado con 1083 archivos
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("=" * 100)
print("🧠 SISTEMA DE CONSULTA INTELIGENTE - CONOCIMIENTO COMPLETO DEL PROYECTO")
print("=" * 100)
print()

# Cargar sistema entrenado
print("📥 Cargando conocimiento entrenado...")
try:
    model = SentenceTransformer("BAAI/bge-m3")
    # Determinar ruta absoluta desde cualquier ubicación
    script_dir = Path(__file__).parent  # scripts/analysis/
    project_root = script_dir.parent.parent  # directorio raíz del proyecto
    data_dir = project_root / "data" / "ml_training"

    index_path = data_dir / "proyecto_COMPLETO_vectordb.index"
    metadata_path = data_dir / "proyecto_COMPLETO_metadata.json"

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadatos_archivos = metadata["metadatos_archivos"]

    print(f"   ✅ Modelo BAAI/bge-m3 cargado")
    print(f"   ✅ Base de datos: {index.ntotal:,} vectores de {index.d} dimensiones")
    print(f"   ✅ Metadata: {len(metadatos_archivos):,} archivos")
    print(f"   📊 Líneas totales: {metadata['total_lineas']:,}")
    print(f"   📄 Caracteres totales: {metadata['total_caracteres']:,}")
    print()

except Exception as e:
    print(f"   ❌ Error cargando sistema: {e}")
    exit(1)


# Función de búsqueda
def buscar(consulta, k=10):
    """Busca archivos relevantes para una consulta"""
    query_emb = (
        model.encode(consulta, convert_to_numpy=True).reshape(1, -1).astype("float32")
    )
    faiss.normalize_L2(query_emb)

    distancias, indices = index.search(query_emb, k)

    resultados = []
    for dist, idx in zip(distancias[0], indices[0]):
        meta = metadatos_archivos[idx]
        resultados.append(
            {
                "archivo": meta["archivo"],
                "similitud": float(dist * 100),
                "extension": meta["extension"],
                "lineas": meta["lineas"],
                "tamaño": meta["tamaño"],
            }
        )

    return resultados


# Estadísticas del proyecto
print("📊 ESTADÍSTICAS DEL PROYECTO:")
print()

archivos_por_tipo = {}
for meta in metadatos_archivos:
    ext = meta["extension"]
    archivos_por_tipo[ext] = archivos_por_tipo.get(ext, 0) + 1

print("   📁 Archivos por tipo:")
for ext, count in sorted(archivos_por_tipo.items(), key=lambda x: x[1], reverse=True)[
    :10
]:
    print(f"      {ext}: {count:,} archivos")

print()

# Top archivos más grandes
print("   📈 Top 10 archivos más grandes:")
archivos_ordenados = sorted(metadatos_archivos, key=lambda x: x["tamaño"], reverse=True)
for i, meta in enumerate(archivos_ordenados[:10], 1):
    print(f"      {i}. {meta['archivo']} - {meta['tamaño']:,} caracteres")

print()
print("=" * 100)
print()

# CONSULTAS AUTOMÁTICAS
print("🔍 EJECUTANDO CONSULTAS DE DEMOSTRACIÓN:")
print()

consultas_demo = [
    ("¿Qué es el MCP Enterprise Master?", 5),
    ("Sistema de aprendizaje automático y RAG", 5),
    ("Auditoría de código y seguridad", 5),
    ("N8N workflows y automatización", 5),
    ("Blockchain y transacciones", 5),
    ("API REST endpoints", 5),
    ("Base de datos vectorial FAISS", 5),
    ("Sistema educativo y cursos", 5),
    ("Docker y contenedores", 5),
    ("Tests y verificación", 5),
]

for consulta, k in consultas_demo:
    print(f"🔎 Consulta: {consulta}")
    resultados = buscar(consulta, k)

    print(f"   📊 Top {k} resultados más relevantes:")
    for i, res in enumerate(resultados, 1):
        print(f"      {i}. {res['archivo']}")
        print(
            f"         Similitud: {res['similitud']:.1f}% | Líneas: {res['lineas']:,} | Tamaño: {res['tamaño']:,} chars"
        )
    print()

print("=" * 100)
print()

# MODO INTERACTIVO
print("💬 MODO INTERACTIVO ACTIVADO")
print("   Escribe tus consultas (o 'salir' para terminar)")
print("=" * 100)
print()

while True:
    try:
        consulta = input("🔎 Tu consulta: ").strip()

        if not consulta:
            continue

        if consulta.lower() in ["salir", "exit", "quit", "q"]:
            print("\n👋 ¡Hasta pronto!")
            break

        print()
        resultados = buscar(consulta, 10)

        print(f"   📊 Top 10 archivos más relevantes:")
        for i, res in enumerate(resultados, 1):
            print(f"      {i}. {res['archivo']} ({res['similitud']:.1f}%)")
            print(
                f"         📝 {res['lineas']:,} líneas | 📄 {res['tamaño']:,} caracteres"
            )
        print()

    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta pronto!")
        break
    except Exception as e:
        print(f"\n   ⚠️  Error: {e}\n")
