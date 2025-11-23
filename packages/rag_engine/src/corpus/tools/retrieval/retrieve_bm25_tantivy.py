import json
from pathlib import Path
from typing import Dict, List

try:
    import tantivy
except Exception as e:
    tantivy = None


def search_tantivy(base: Path, query: str, top_k: int = 10, fields=None) -> List[Dict]:
    if tantivy is None:
        return []
    idx_dir = base / "index" / "tantivy"
    if not idx_dir.exists():
        return []
    index = tantivy.Index.open(idx_dir)
    schema = index.schema
    reader = index.reader()
    searcher = reader.searcher()
    if not fields:
        fields = ["title", "text"]
    fn = [schema.get_field(f) for f in fields if schema.get_field(f) is not None]
    if not fn:
        return []
    qp = tantivy.QueryParser(schema, fn)
    q = qp.parse_query(query)
    hits = searcher.search(q, top_k).hits
    results = []
    for score, addr in hits:
        doc = searcher.doc(addr)
        doc = {k: v[0] if isinstance(v, list) and v else v for k, v in doc.items()}
        results.append(
            {
                "score": float(score),
                "chunk_id": doc.get("chunk_id", ""),
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "meta": {
                    "lang": doc.get("lang", ""),
                    "tags": doc.get("tags", "").split(",") if doc.get("tags") else [],
                },
            }
        )
    return results
