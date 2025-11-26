from pathlib import Path

import yaml

from tools.retrieval.retrieve_bm25_tantivy import search_tantivy
from tools.retrieval.retrieve_bm25_tantivy_sharded import search_tantivy_sharded
from tools.retrieval.search_bm25_whoosh import search_bm25 as search_whoosh


def _cfg():
    with open("config/universal.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def lexical_search(base: Path, query: str, top_k: int = 10, fields=None):
    cfg = _cfg()
    lex = cfg.get("retrieval", {}).get("lexical", {})
    backend = lex.get("backend", "whoosh").lower()
    shards = bool(lex.get("shards", {}).get("enabled", False))
    if backend == "tantivy":
        if shards:
            res = search_tantivy_sharded(base, query, top_k=top_k)
            if res:
                return res
        else:
            res = search_tantivy(
                base, query, top_k=top_k, fields=fields or ["title", "text"]
            )
            if res:
                return res
    return search_whoosh(base, query, top_k=top_k)
