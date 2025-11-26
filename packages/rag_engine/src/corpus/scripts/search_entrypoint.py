#!/usr/bin/env python
"""
Minimal search entrypoint that pre-imports torch to avoid Windows DLL reload issues.
Run directly as: python scripts/search_entrypoint.py "query"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch  # Pre-import torch early to stabilize DLL on Windows

from tools.retrieval.search_local_hnsw import search_local_hnsw


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/search_entrypoint.py <query> [top_k]")
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    results = search_local_hnsw(Path("."), query, top_k=top_k)

    if not results:
        print("No results found.")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['score']:.4f}")
            print(f"    Doc: {r['doc_id']}")
            print(f"    Text: {r['text'][:150]}...")


if __name__ == "__main__":
    main()
