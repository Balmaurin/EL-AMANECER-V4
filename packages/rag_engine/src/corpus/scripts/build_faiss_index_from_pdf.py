import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=512, overlap=64):
    # Simple sliding window chunking
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def main():
    base = Path(".")
    pdf_path = base / "CiberseguridadenelMC.pdf"
    index_dir = base / "index"
    index_dir.mkdir(exist_ok=True)
    mapping_path = index_dir / "mapping.parquet"
    faiss_index_path = index_dir / "faiss.index"

    print(f"Leyendo PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    print(f"Texto extraído, longitud: {len(text)}")
    chunks = chunk_text(text)
    print(f"Total de chunks: {len(chunks)}")

    model = SentenceTransformer("BAAI/bge-m3")
    print("Generando embeddings...")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Crear índice FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(faiss_index_path))
    print(f"Índice FAISS guardado en: {faiss_index_path}")

    # Crear mapping
    import json

    df = pd.DataFrame(
        {
            "chunk_id": [str(i) for i in range(len(chunks))],
            "doc_id": ["CiberseguridadenelMC.pdf"] * len(chunks),
            "title": ["Ciberseguridad en el MC"] * len(chunks),
            "text": chunks,
            "meta": [json.dumps({}) for _ in chunks],
        }
    )
    df.to_parquet(mapping_path, index=False)
    print(f"Mapping guardado en: {mapping_path}")


def build_faiss_index_from_pdf(pdf_path: str, index_dir: str):
    pdf_path = Path(pdf_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(exist_ok=True)
    mapping_path = index_dir / f"{pdf_path.stem}_mapping.parquet"
    faiss_index_path = index_dir / f"{pdf_path.stem}_faiss.index"

    print(f"Leyendo PDF: {pdf_path}")
    text = extract_text_from_pdf(str(pdf_path))
    print(f"Texto extraído, longitud: {len(text)}")
    chunks = chunk_text(text)
    print(f"Total de chunks: {len(chunks)}")

    model = SentenceTransformer("BAAI/bge-m3")
    print("Generando embeddings...")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(faiss_index_path))
    print(f"Índice FAISS guardado en: {faiss_index_path}")

    import json

    df = pd.DataFrame(
        {
            "chunk_id": [str(i) for i in range(len(chunks))],
            "doc_id": [pdf_path.name] * len(chunks),
            "title": [pdf_path.stem] * len(chunks),
            "text": chunks,
            "meta": [json.dumps({}) for _ in chunks],
        }
    )
    df.to_parquet(mapping_path, index=False)
    print(f"Mapping guardado en: {mapping_path}")


if __name__ == "__main__":
    base = Path(".")
    pdf_path = base / "CiberseguridadenelMC.pdf"
    index_dir = base / "index"
    build_faiss_index_from_pdf(str(pdf_path), str(index_dir))
