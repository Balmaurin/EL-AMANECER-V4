import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def search_tantivy_sharded(base: Path, query: str, top_k: int = 10) -> List[Dict]:
    try:
        import tantivy
    except Exception:
        return []
    shard_dir = base / "index" / "tantivy_shards"
    if not shard_dir.exists():
        return []
    results = []
    for sdir in sorted(shard_dir.glob("shard_*")):
        try:
            idx = tantivy.Index.open(sdir)
            searcher = idx.searcher()
            qp = tantivy.QueryParser(idx.schema(), ["title", "text"])
            q = qp.parse_query(query)
            hits = searcher.search(q, top_k).hits
            for score, addr in hits:
                doc = searcher.doc(addr)
                doc = {
                    k: v[0] if isinstance(v, list) and v else v for k, v in doc.items()
                }
                results.append(
                    {
                        "score": float(score),
                        "chunk_id": doc.get("chunk_id", ""),
                        "doc_id": doc.get("doc_id", ""),
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                    }
                )
        except Exception as e:
            logger.debug(f"Tantivy shard search failed in {sdir}: {e}")
            continue
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]
