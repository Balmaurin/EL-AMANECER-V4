import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_jsonl(file_path: str) -> str:
    import json

    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "text" in obj:
                    texts.append(str(obj["text"]))
                else:
                    texts.append(str(obj))
            except Exception as e:
                log.debug(f"Skipping malformed JSON line: {e}")
                continue
    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def build_corpus_index(data_dir: str, index_dir: str):
    data_dir = Path(data_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(exist_ok=True)
    all_chunks = []
    all_embeddings = []
    all_meta = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando modelo de embeddings en {device}...")
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    for file in data_dir.iterdir():
        ext = file.suffix.lower()
        try:
            if ext == ".pdf":
                print(f"Procesando PDF: {file.name}")
                text = extract_text_from_pdf(str(file))
            elif ext == ".txt":
                print(f"Procesando TXT: {file.name}")
                text = extract_text_from_txt(str(file))
            elif ext == ".md":
                print(f"Procesando MD: {file.name}")
                text = extract_text_from_md(str(file))
            elif ext == ".jsonl":
                print(f"Procesando JSONL: {file.name}")
                text = extract_text_from_jsonl(str(file))
            else:
                continue
        except FileNotFoundError:
            print(f"Archivo no encontrado o eliminado: {file.name}, ignorando...")
            continue
        except Exception as e:
            print(f"Error procesando {file.name}: {e}, ignorando...")
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)
        all_meta.extend([{"doc_id": file.name, "title": file.stem}] * len(chunks))
    if not all_chunks:
        print("No se encontraron archivos válidos o chunks.")
        return
    all_embeddings = np.vstack(all_embeddings)
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(all_embeddings.astype(np.float32))
    faiss_index_path = index_dir / "corpus_faiss.index"
    faiss.write_index(index, str(faiss_index_path))
    print(f"Índice FAISS global guardado en: {faiss_index_path}")
    df = pd.DataFrame(
        {
            "chunk_id": [str(i) for i in range(len(all_chunks))],
            "doc_id": [meta["doc_id"] for meta in all_meta],
            "title": [meta["title"] for meta in all_meta],
            "text": all_chunks,
            "meta": [json.dumps({}) for _ in all_chunks],
        }
    )
    mapping_path = index_dir / "corpus_mapping.parquet"
    df.to_parquet(mapping_path, index=False)
    print(f"Mapping global guardado en: {mapping_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Construye índice y mapping global del corpus (PDF, TXT, MD)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/", help="Carpeta con archivos de datos"
    )
    parser.add_argument(
        "--index_dir", type=str, default="index/", help="Carpeta de salida para índices"
    )
    args = parser.parse_args()
    build_corpus_index(args.data_dir, args.index_dir)
