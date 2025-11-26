import json
import shutil
from pathlib import Path
from typing import Dict, List

from whoosh import index
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.qparser import QueryParser


class WhooshSearcher:
    """Clase para realizar búsquedas usando Whoosh con BM25."""

    def __init__(self, base_path: Path):
        """
        Inicializa el buscador Whoosh.

        Args:
            base_path: Directorio base del índice
        """
        self.ix_dir = base_path / "index" / "bm25"
        self.schema = Schema(
            chunk_id=ID(stored=True, unique=True),
            doc_id=ID(stored=True),
            title=TEXT(analyzer=StandardAnalyzer(), stored=True),
            lang=ID(stored=True),
            tags=TEXT(stored=True),
            text=TEXT(analyzer=StandardAnalyzer(), stored=True),
        )

        if not self.ix_dir.exists() or not index.exists_in(self.ix_dir):
            raise RuntimeError("No se encontró el índice Whoosh en " + str(self.ix_dir))

        self.ix = index.open_dir(self.ix_dir)
        self.parser = QueryParser("text", self.ix.schema)

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Realiza una búsqueda usando BM25.

        Args:
            query: Query de búsqueda
            limit: Número máximo de resultados

        Returns:
            Lista de documentos encontrados con sus scores
        """
        with self.ix.searcher() as searcher:
            query = self.parser.parse(query)
            results = searcher.search(query, limit=limit)
            return [
                {
                    "chunk_id": hit["chunk_id"],
                    "doc_id": hit["doc_id"],
                    "title": hit["title"],
                    "text": hit["text"],
                    "score": hit.score,
                    "rank": i + 1,
                }
                for i, hit in enumerate(results)
            ]


def build_bm25(base: Path):
    """Construye un índice BM25 usando Whoosh."""
    ix_dir = base / "index" / "bm25"
    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        doc_id=ID(stored=True),
        title=TEXT(analyzer=StandardAnalyzer(), stored=True),
        lang=ID(stored=True),
        tags=TEXT(stored=True),
        text=TEXT(analyzer=StandardAnalyzer(), stored=True),
    )
    if ix_dir.exists() and index.exists_in(ix_dir):
        ix = index.open_dir(ix_dir)
        if ix.schema != schema:
            shutil.rmtree(ix_dir)
            ix_dir.mkdir(parents=True, exist_ok=True)
            ix = index.create_in(ix_dir, schema)
    else:
        if ix_dir.exists():
            shutil.rmtree(ix_dir)
        ix_dir.mkdir(parents=True, exist_ok=True)
        ix = index.create_in(ix_dir, schema)

    writer = ix.writer(limitmb=512, procs=1, multisegment=True)
    for jf in (base / "chunks").glob("*.jsonl"):
        rec = json.loads(jf.read_text(encoding="utf-8"))
        meta = rec.get("meta", {})
        writer.update_document(
            chunk_id=rec["chunk_id"],
            doc_id=rec["doc_id"],
            title=meta.get("title", ""),
            lang=meta.get("lang", ""),
            tags=",".join(meta.get("tags", [])),
            text=rec["text"],
        )
    writer.commit()
    print("[OK] BM25 (Whoosh) incremental listo")
