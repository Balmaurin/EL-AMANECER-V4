import os
from pathlib import Path

from scripts.build_faiss_index_from_pdf import build_faiss_index_from_pdf


def ingest_all_pdfs(pdf_dir, index_dir):
    pdf_dir = Path(pdf_dir)
    index_dir = Path(index_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No se encontraron PDFs en {pdf_dir}")
        return
    for pdf in pdf_files:
        print(f"Procesando: {pdf.name}")
        build_faiss_index_from_pdf(str(pdf), str(index_dir))
    print("Ingesta completada.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingesta masiva de PDFs a FAISS")
    parser.add_argument("--pdf_dir", type=str, default="data/", help="Carpeta con PDFs")
    parser.add_argument(
        "--index_dir", type=str, default="index/", help="Carpeta de salida para índices"
    )
    args = parser.parse_args()
    ingest_all_pdfs(args.pdf_dir, args.index_dir)
